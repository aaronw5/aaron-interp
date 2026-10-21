"""P37 — top-quark (t->bqq') head circuit in ParT and an 80-head x 10-class knock-out map (loop C round C9, exploratory; claim C17).

Question. Is ParT's t->bqq' call carried by a few attention heads (so that keeping only those heads keeps the top call far above
random head sets of the same size), or is it spread over many heads? Is the set of heads each class call depends on sparse and
class-specific (low overlap between the top, W/Z and lepton head sets), or shared?

Models: ParT_full, ParT_kin (released weights, models/part); two nulls built from ParT_full: 'untrained' (every weight redrawn from
a Gaussian with the trained tensor's mean and spread, seed 37) and 'perm' (output rows of every weight matrix permuted, matching
biases permuted alike, P33 recipe, seed 3725).
Sample: JetClass test_20M <sample>_111.root (TTBar and 7 other classes never used before; W/Z rows of _111 used by P13v2, 3,000
of 100,000 per class), numpy default_rng(3725), random rows (never first-k), split per class into disjoint sets:
  EVAL  1,000 jets per class (N 10,000): accuracy, the 80 x 10 knock-out matrix, literal circuits, edits;
  REF     200 per class (N 2,000): the mean head outputs and MLP outputs used for mean ablation;
  REF2    200 per class (N 2,000): a second mean, the noise floor of the ablation (same ablation, mean from other random jets);
  SCAN  1,000 further TTBar (t->bqq') jets: the per-head top scan used to rank heads (ranking and circuit evaluation use
        different jets).
  P1 reference: rows 3,000-4,000 of each class file _100 (part_core.sample), known accuracy ParT_full 85.93 %, ParT_kin 74.47 %.
Every jet passes once through part_core.build (eta = jet_eta + sign(jet_eta) deta).
Heads: 8 particle blocks (p1..p8) and 2 class-attention blocks (b1, b2), 8 heads each = 80 heads. A head's output is its
contribution to the attention module output, sum_h (A_h V_h) W_O[h] (check: sum over heads + out_proj bias = module output, logged
as recon_max). Mean ablation of a head = its output at every token replaced by its mean over the REF set's real tokens (class
blocks: the class token). Mean ablation of an MLP = the fc2 output of that block replaced by its REF mean.
Step 1a (top scan, SCAN jets): per head, mean-ablate it alone; per jet d_QCD = Delta(logit_t - logit_QCD) = Delta log(p_t/p_QCD),
  d_W = Delta(logit_t - logit_W). Mean, 95 % bootstrap interval (1,000 resamples of jets), sign-flip permutation p (20,000 flips;
  H0: ablation leaves the contrast unchanged on average), Holm over 80 heads x 2 contrasts = 160 tests. Noise floor: the same
  ablation with the REF2 mean. Joint knock-outs: the top-k ranked heads ablated together vs all 80 together (share of the drop)
  and vs 20 random k-sets.
Step 1b (80 x 10 matrix, EVAL jets): per head, mean-ablate it alone; per class k, change of recall (fraction of true-k jets called
  k), bootstrap interval over the class's jets, McNemar on the paired correct/incorrect calls, Benjamini-Hochberg over 800 tests;
  and the mean change of the class's own log-odds lo_k = log p_k/(1-p_k) on its own jets. Class head set = heads with BH q < 0.05
  and a recall decrease; also fixed-size sets (5 heads with the most negative own-class Delta lo_k). Jaccard overlaps.
  Noise floor (ParT_full): the whole matrix again with the REF2 mean.
Step 2 (literal circuits, EVAL): heads ranked by the mean of their two top scan effects (most negative first; Holm-significant
  negative heads first). Circuit k = the first k heads. Variant 'literal' (user definition): only the circuit heads kept, every
  other head AND all 10 MLPs mean-ablated (embeddings, pair embedding, LayerNorms, class token and final linear layer kept).
  Variant 'cls_mlp': as literal but the two class-block MLPs kept (P34: without them the lepton circuit gave a constant call).
  Variant 'heads_only': other heads mean-ablated, all MLPs kept. k = 0, 1, 2, 3, 5, 10, 20, 40, 80.
  Metrics: t->bqq' recall, t-vs-QCD accuracy (logit_t > logit_QCD on true t and QCD jets), t-vs-QCD AUC (score logit_t -
  logit_QCD), class-balanced 10-class accuracy, 10-class accuracy, confusion. Null: NRAND random head sets of the same size
  (same sets for the three variants), on a random 100-per-class subset SUB of EVAL; percentile p = (1 + #random >= circuit) /
  (1 + NRAND).
Step 3 (input edit, EVAL true t->bqq' jets): exclusive kT clustering (d_ij = min(pT_i^2, pT_j^2) DR_ij^2, E-scheme, rapidity-phi,
  no beam distance) down to 3 subjets; W pair = the subjet pair with invariant mass nearest 80.4 GeV; b-prong = the third subjet.
  Displacement check: the subjet holding the track with the largest transverse impact-parameter significance |d0|/sigma_d0.
  edit a: the b-prong's particles replaced by one particle at their four-momentum sum (slot of the hardest of them; its charge /
          PID / impact-parameter features kept). Jet four-momentum unchanged (the merged particle carries the prong's mass).
  edit b (control): a cluster of W-prong particles with the same summed pT merged the same way.
  edit c (control): one whole W subjet (of the two, the one with pT closest to the b-prong's) merged the same way.
  edit b2 (control): W-prong cluster with the b-prong's particle count (full-network change only).
  Full change: Delta lo_t and Delta(logit_t - logit_QCD), edited - unedited. Patching: the unedited jet's run with the top-k heads'
  outputs replaced by their outputs in the edited run (tokens aligned by particle; removed tokens keep their unedited outputs);
  share = mean patched change / mean full change; null: NRAND random head sets of the same size.
Step 4 (head description, ParT_full, SCAN jets): attention weights of each head: share of attention onto the b-prong, the W
  prongs, charged hadrons, neutral hadrons, photons, leptons, the leading particle, the class token itself; mean DR of the attended
  particle to the jet axis (class heads) or to the query particle (particle heads); same-prong share; effective number of keys.
  ./sync.sh && ./part_job.sh c-p37-run 'python -m interp.p37_top_circuit --models full,untrained,perm'  (and --models kin)
  python -m interp.p37_top_circuit --plot  -> results/part/p37_top_circuit.png
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--plot', action='store_true')
ap.add_argument('--models', default='full,kin,untrained,perm'); ap.add_argument('--nrand', type=int, default=None)
ARGS = ap.parse_args()
SMOKE = ARGS.smoke
SEED = 3725
FILE_NO = '111'
NEVAL = 30 if SMOKE else 1000
NREF = 20 if SMOKE else 200
NSCAN = 60 if SMOKE else 1000
NSUB = 10 if SMOKE else 100
NB = 200 if SMOKE else 1000
NPERM = 2000 if SMOKE else 20000
NRAND = ARGS.nrand or (3 if SMOKE else 100)
NRJ = 3 if SMOKE else 20
KS = [0, 1, 2, 3, 5, 10, 20, 40, 80]
KPATCH = [5, 10]
BS = 250
TEST = Path('/j-jepa-vol/JetClass/Pythia/test_20M')
OUTD = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
TAG = '_smoke' if SMOKE else ''
QCD, ZQQ, WQQ, TOP, TBL, HQQL = 0, 6, 7, 8, 9, 5
CLN = ['QCD', 'H->bb', 'H->cc', 'H->gg', 'H->4q', "H->lvqq'", 'Z->qq', "W->qq'", "t->bqq'", 't->blv']
C, H, HD = 128, 8, 16
MW = 80.4
MODEL_SPEC = dict(full=('ParT_full', 'full', None), kin=('ParT_kin', 'kin', None), untrained=('ParT_full_untrained', 'full', 'untrained'),
                  perm=('ParT_full_perm', 'full', 'perm'))
T0 = time.time()


def fn_for(key):
    return OUTD / f'p37_top_circuit_{key}{TAG}.json'


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


def bh(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); q = np.empty(m); run = 1.0
    for r in range(m - 1, -1, -1):
        i = o[r]; run = min(run, p[i] * m / (r + 1)); q[i] = run
    return q


def signflip_p(x, S):
    """two-sided sign-flip permutation p of mean(x) = 0 (S: (nperm, n) random +-1)"""
    x = np.asarray(x, np.float64); obs = abs(x.mean()); null = np.abs(S @ x) / len(x)
    return float((1 + (null >= obs - 1e-12).sum()) / (1 + len(S)))


def mstat(x, W, S=None):
    x = np.asarray(x, np.float64); bs = W.astype(np.float64) @ x
    e = dict(v=float(x.mean()), ci=ci(bs), se=float(bs.std()))
    if S is not None: e['p_perm'] = signflip_p(x, S)
    return e


def mcnemar_counts(a, b):
    a = np.asarray(a, bool); b = np.asarray(b, bool); return int((a & ~b).sum()), int((~a & b).sum())


def mcnemar_p(n10, n01):
    from scipy.stats import binomtest
    return float(binomtest(min(n10, n01), n10 + n01, 0.5).pvalue) if n10 + n01 > 0 else 1.0


def mcnemar(a, b):
    n10, n01 = mcnemar_counts(a, b)
    return dict(before=float(np.mean(a)), after=float(np.mean(b)), lost=n10, gained=n01, p=mcnemar_p(n10, n01))


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


def auc(sp, sn):
    from scipy.stats import rankdata
    sp, sn = np.asarray(sp, float), np.asarray(sn, float)
    if len(sp) == 0 or len(sn) == 0: return None
    r = rankdata(np.concatenate([sp, sn])); return float((r[:len(sp)].sum() - len(sp) * (len(sp) + 1) / 2) / (len(sp) * len(sn)))


def acc_block(L, y, W=None, conf=True):
    c = L.argmax(1); rec = [float((c[y == k] == k).mean()) if (y == k).any() else None for k in range(10)]
    tq = np.isin(y, [QCD, TOP]); s = ctr(L, TOP, QCD); bin_ok = ((s > 0) == (y == TOP))[tq]
    e = dict(acc10=float((c == y).mean()), balanced=float(np.mean([r for r in rec if r is not None])), recall=rec,
             top_recall=rec[TOP], bin_tq=float(bin_ok.mean()), auc_tq=auc(s[y == TOP], s[y == QCD]),
             top_call_rate=float((c == TOP).mean()), n_distinct_calls=int(len(np.unique(c))))
    if conf: e['confusion'] = [[int(((y == i) & (c == j)).sum()) for j in range(10)] for i in range(10)]
    if W is not None: e['acc10_ci'] = ci(W @ (c == y).astype(np.float32))
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
    """per jet: 3 exclusive-kT subjets; W pair (mass nearest 80.4 GeV); b-prong members; pT-matched and count-matched W-prong
    clusters; the W subjet with pT closest to the b-prong's; the subjet holding the most displaced track (|d0|/sigma_d0)"""
    N = len(B['mask']); info = dict(gb=[None] * N, gw=[None] * N, gw2=[None] * N, gc=[None] * N, mjet=np.full(N, np.nan),
                                   mW=np.full(N, np.nan), nb=np.zeros(N, int), nw=np.zeros(N, int), nc=np.zeros(N, int),
                                   ptb=np.full(N, np.nan), ptw=np.full(N, np.nan), ptc=np.full(N, np.nan), tau32=np.full(N, np.nan),
                                   lab=[None] * N, disp_is_b=np.full(N, np.nan), sig_max_b=np.full(N, np.nan), sig_max_w=np.full(N, np.nan),
                                   ndisp_b=np.full(N, np.nan), ndisp_w=np.full(N, np.nan), ndisp_is_b=np.full(N, np.nan))
    for j in range(N):
        p4 = p4_of(B, j); n = len(p4)
        if n < 6: continue
        o = kt_exclusive(p4, (3, 2)); memb, S3 = o[3]
        pairs = [(0, 1), (0, 2), (1, 2)]; mm = [m_of(S3[list(pq)]) for pq in pairs]; w = int(np.argmin(np.abs(np.array(mm) - MW)))
        bi = 3 - sum(pairs[w]); gb = memb[bi]; wm = memb[pairs[w][0]] + memb[pairs[w][1]]
        info['mjet'][j] = m_of(p4); info['mW'][j] = mm[w]; info['tau32'][j] = tau(p4, S3) / max(tau(p4, o[2][1]), 1e-12)
        lab = np.zeros(n, int); lab[memb[pairs[w][0]]] = 1; lab[memb[pairs[w][1]]] = 2; info['lab'][j] = lab
        pt = np.hypot(p4[:, 0], p4[:, 1]); ptb = pt[gb].sum(); info['ptb'][j] = ptb; info['nb'][j] = len(gb)
        # displacement: largest |d0|/sigma_d0 among charged particles of each subjet
        chg = np.abs(B['charge'][j, :n]) > 0; sig = np.where(chg, np.abs(B['d0val'][j, :n]) / np.maximum(B['d0err'][j, :n], 1e-6), 0.0)
        smax = [float(sig[m_].max()) if len(m_) else 0.0 for m_ in memb]
        if max(smax) > 0: info['disp_is_b'][j] = float(int(np.argmax(smax)) == bi)
        d0a = np.abs(B['d0val'][j, :n]); nd = [int(((sig[m_] > 3) & (d0a[m_] < 2.0)).sum()) for m_ in memb]
        info['ndisp_b'][j] = nd[bi]; info['ndisp_w'][j] = max(nd[pairs[w][0]], nd[pairs[w][1]])
        if max(nd) > 0 and sorted(nd)[-1] > sorted(nd)[-2]: info['ndisp_is_b'][j] = float(int(np.argmax(nd)) == bi)
        info['sig_max_b'][j] = smax[bi]; info['sig_max_w'][j] = max(smax[pairs[w][0]], smax[pairs[w][1]])
        if len(gb) >= 2: info['gb'][j] = gb
        # control c: the W subjet with pT closest to the b-prong's
        wsub = [memb[pairs[w][0]], memb[pairs[w][1]]]; wpt = [pt[s_].sum() for s_ in wsub]; c_ = int(np.argmin(np.abs(np.array(wpt) - ptb)))
        if len(wsub[c_]) >= 2: info['gc'][j] = wsub[c_]; info['nc'][j] = len(wsub[c_]); info['ptc'][j] = wpt[c_]
        wm = np.asarray(wm); seed = rng.choice(wm); _, y, ph = _ypp(p4)
        dr = np.hypot(y[wm] - y[seed], _wrap(ph[wm] - ph[seed])); order = wm[np.argsort(dr, kind='stable')]
        cs = np.cumsum(pt[order])
        if cs[-1] >= ptb:
            q = int(np.searchsorted(cs, ptb)) + 1
            if q >= 2 and q < len(wm) + 1: info['gw'][j] = order[:q].tolist(); info['nw'][j] = q; info['ptw'][j] = cs[q - 1]
        if len(gb) >= 2 and len(order) >= len(gb): info['gw2'][j] = order[:len(gb)].tolist()
    return info


# ------------------------------------------------------------------------------------------------ ParT with head / MLP controls
class HRun:
    """ParT with per-head mean ablation, per-head donor replacement at aligned tokens, MLP (fc2 output) mean ablation, and the
    recording of means / head outputs / attention weights. Attention modules 0-7 = particle blocks p1..p8, 8-9 = class blocks b1,
    b2. ctl per batch: mean {i: (sel (H,) float, M (H, C))}; mlp {i: Mf (C,)}; rep {i: (sel (H,), D (n, H, Lq, C), pos (n, Lq))};
    recmean True; recch set(i); recw set(i)"""

    def __init__(self, feat, null=None):
        import torch
        from interp.part_core import ParTNetwork
        self.t, self.feat = torch, feat
        self.net = ParTNetwork(feat, untrained=(null == 'untrained'), seed=37); self.m = self.net.model; self.dev = self.net.device
        if null == 'perm': self.nperm = self._permute(SEED)
        self.m.eval()
        self.ctl = {}; self.out = {}; self.recon = 0.0; self.curq = {}
        self.reset_means()
        blocks = list(self.m.blocks) + list(self.m.cls_blocks)
        for i, b in enumerate(blocks):
            self._wrap(i, b.attn); b.fc2.register_forward_hook(lambda mod, inp, out, i=i: self._mlp(i, out))

    def _permute(self, seed):
        """P33 recipe: output rows of every weight matrix permuted (one permutation per tensor), biases of the same module alike"""
        t = self.t; g = t.Generator().manual_seed(seed); sd = self.m.state_dict(); new = {}; perms = {}
        for k, v in sd.items():
            if k.endswith('weight') and v.ndim >= 2 and v.is_floating_point():
                p = t.randperm(v.shape[0], generator=g).to(v.device); perms[k[:-len('weight')]] = p; new[k] = v[p].clone()
        for k, v in sd.items():
            if k in new: continue
            pre = k[:-len('bias')] if k.endswith('bias') else None
            if pre is not None and pre in perms and v.ndim == 1 and v.shape[0] == len(perms[pre]): new[k] = v[perms[pre]].clone()
            elif k.endswith('in_proj_bias') and (k[:-len('in_proj_bias')] + 'in_proj_') in perms:
                new[k] = v[perms[k[:-len('in_proj_bias')] + 'in_proj_']].clone()
            else: new[k] = v
        self.m.load_state_dict(new); return len(perms)

    def reset_means(self):
        t = self.t
        self.msum = {i: t.zeros(H, C, dtype=t.float64, device=self.dev) for i in range(10)}; self.mcnt = {i: 0.0 for i in range(10)}
        self.fsum = {i: t.zeros(C, dtype=t.float64, device=self.dev) for i in range(10)}; self.fcnt = {i: 0.0 for i in range(10)}

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
            c = S.ctl; mean = c.get('mean', {}).get(i); rep = c.get('rep', {}).get(i); recm = c.get('recmean')
            recch = i in c.get('recch', ()); recw = i in c.get('recw', ())
            Lq, n, _ = query.shape; Sk = key.shape[0]
            realq = (~key_padding_mask).float() if i < 8 else t.ones(n, 1, device=query.device)
            S.curq[i] = realq
            if mean is None and rep is None and not recm and not recch and not recw:
                return orig(query, key, value, key_padding_mask=key_padding_mask, need_weights=False, attn_mask=attn_mask)
            out, w = orig(query, key, value, key_padding_mask=key_padding_mask, need_weights=True, attn_mask=attn_mask, average_attn_weights=False)
            if recw: S.out.setdefault('w', {})[i] = w.detach()
            if mean is None and rep is None and not recm and not recch: return out, None
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
        self.ctl = {}; self.out = {}; return L


HEADS = [(i, h) for i in range(10) for h in range(H)]


def hlab(ih):
    i, h = ih if isinstance(ih, tuple) else divmod(int(ih), H); return f'p{i + 1}h{h + 1}' if i < 8 else f'b{i - 7}h{h + 1}'


def ctl_keep(M, keep, Mh, Mf, mlp_idx):
    """keep: head indices 0..79 kept; every other head mean-ablated; MLPs in mlp_idx mean-ablated"""
    t = M.t; ks = set(int(x) for x in keep); mean = {}
    for i in range(10):
        sel = t.tensor([0.0 if 8 * i + h in ks else 1.0 for h in range(H)], device=M.dev)
        if float(sel.sum()) > 0: mean[i] = (sel, Mh[i])
    return dict(mean=mean, mlp={i: Mf[i] for i in mlp_idx})


def ctl_ablate(M, heads, Mh):
    t = M.t; mean = {}
    for k in heads:
        i, h = divmod(int(k), H)
        if i not in mean: mean[i] = (t.zeros(H, device=M.dev), Mh[i])
        mean[i][0][h] = 1.0
    return dict(mean=mean)


VARIANTS = dict(literal=list(range(10)), cls_mlp=list(range(8)), heads_only=[])


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


# ------------------------------------------------------------------------------------------------ head description
CATS = ['b-prong', 'W prongs', 'charged hadrons', 'neutral hadrons', 'photons', 'leptons', 'leading particle']


def describe(M, Bd, labs, bs=50):
    """attention statistics of every head on jets Bd (true top jets with kT prong labels labs[j] (n_j,), 0 = b-prong)"""
    from interp.part_core import sub, inputs
    t = M.t; N = len(Bd['mask']); order = np.argsort(Bd['mask'].sum(1), kind='stable')
    acc = {k: np.zeros((10, H)) for k in ('cls_self', 'dr', 'same', 'self', 'neff', 'n')}
    acc['cat'] = np.zeros((10, H, len(CATS))); base = np.zeros(len(CATS)); basept = np.zeros(len(CATS)); nj = 0
    for s in range(0, N, bs):
        idx = order[s:s + bs]; Jb = sub(Bd, idx); x, _, _ = inputs(Jb, M.feat); P = x.shape[-1]; n = len(idx)
        msk = Jb['mask'][:, :P] > 0; lab = np.full((n, P), -1)
        for q, j in enumerate(idx): lab[q, :len(labs[j])] = labs[j]
        cat_ = np.stack([lab == 0, lab > 0, (np.abs(Jb['charge'][:, :P]) > 0) & (Jb['isElectron'][:, :P] + Jb['isMuon'][:, :P] == 0),
                         Jb['isNeutralHadron'][:, :P] > 0, Jb['isPhoton'][:, :P] > 0, (Jb['isElectron'][:, :P] + Jb['isMuon'][:, :P]) > 0,
                         np.broadcast_to(np.arange(P)[None] == 0, (n, P))], -1) & msk[..., None]
        pt = np.hypot(Jb['px'][:, :P], Jb['py'][:, :P]) * msk; nr = msk.sum(1)
        base += (cat_.sum(1) / nr[:, None]).sum(0); basept += ((cat_ * pt[..., None]).sum(1) / pt.sum(1)[:, None]).sum(0); nj += n
        dra = np.hypot(Jb['deta'][:, :P], Jb['dphi'][:, :P]) * msk
        drq = np.hypot(Jb['deta'][:, :P, None] - Jb['deta'][:, None, :P], _wrap(Jb['dphi'][:, :P, None] - Jb['dphi'][:, None, :P]))
        same = (lab[:, :, None] == lab[:, None, :]) & (lab[:, :, None] >= 0)
        tc = lambda a: t.from_numpy(np.asarray(a, np.float32)).to(M.dev)
        CAT, DRA, DRQ, SAME, MQ = tc(cat_), tc(dra), tc(drq), tc(same), tc(msk)
        M.ctl = dict(recw=set(range(10))); M.out = {}; M.fwd(Jb); Wd = M.out['w']; M.ctl = {}
        for i in range(10):
            w = Wd[i].float()
            if i >= 8:
                a = w[:, :, 0, :]; acc['cls_self'][i] += a[..., 0].sum(0).cpu().numpy(); ap_ = a[..., 1:1 + P]
                acc['cat'][i] += t.einsum('nhp,npk->hk', ap_, CAT).cpu().numpy(); acc['dr'][i] += t.einsum('nhp,np->h', ap_, DRA).cpu().numpy()
                acc['neff'][i] += t.exp(-(a * t.log(a.clamp_min(1e-12))).sum(-1)).sum(0).cpu().numpy()
            else:
                nq = MQ.sum(1)[:, None, None]; wq = w * MQ[:, None, :, None]
                aq = wq.sum(2) / nq                                                                   # (n, H, P) mean over real queries
                acc['cat'][i] += t.einsum('nhp,npk->hk', aq, CAT).cpu().numpy()
                acc['dr'][i] += ((wq * DRQ[:, None]).sum((2, 3)) / nq[..., 0]).sum(0).cpu().numpy()
                acc['same'][i] += ((wq * SAME[:, None]).sum((2, 3)) / nq[..., 0]).sum(0).cpu().numpy()
                acc['self'][i] += ((wq * t.eye(P, device=M.dev)[None, None]).sum((2, 3)) / nq[..., 0]).sum(0).cpu().numpy()
                ent = t.exp(-(w * t.log(w.clamp_min(1e-12))).sum(-1))                                # (n, H, P)
                acc['neff'][i] += ((ent * MQ[:, None]).sum(2) / nq[..., 0]).sum(0).cpu().numpy()
        del Wd; M.out = {}
    out = {}; base /= nj; basept /= nj
    for k in range(80):
        i, h = divmod(k, H); sh = acc['cat'][i, h] / nj
        e = dict(share={CATS[q]: float(sh[q]) for q in range(len(CATS))}, dr=float(acc['dr'][i, h] / nj), neff=float(acc['neff'][i, h] / nj))
        if i >= 8: e['cls_self'] = float(acc['cls_self'][i, h] / nj)
        else: e['same_prong'] = float(acc['same'][i, h] / nj); e['self'] = float(acc['self'][i, h] / nj)
        e['text'] = head_text(k, e, base); out[hlab(k)] = e
    return out, dict(uniform={CATS[q]: float(base[q]) for q in range(len(CATS))}, pt_share={CATS[q]: float(basept[q]) for q in range(len(CATS))}, n_jets=nj)


def head_text(k, e, base):
    i, h = divmod(k, H); sh = e['share']
    rat = {c: sh[c] / max(base[q], 1e-3) for q, c in enumerate(CATS) if sh[c] >= 0.10}
    best = max(rat, key=rat.get) if rat else None
    if i >= 8:
        q = f'class-token attention ({"block 1" if i == 8 else "block 2"})'
        if e.get('cls_self', 0) > 0.5: return f'{hlab(k)}: {q} mostly onto the class token itself ({e["cls_self"]:.0%})'
        tgt = f'onto the {best} ({sh[best]:.0%} of weight vs {base[CATS.index(best)]:.0%} of particles)' if best and rat[best] > 1.5 else 'spread over the particles'
        return f'{hlab(k)}: {q} {tgt}, mean dR to axis {e["dr"]:.2f}, {e["neff"]:.1f} effective keys'
    q = f'particle-block-{i + 1} attention'
    tgt = f'onto {best} ({sh[best]:.0%} vs {base[CATS.index(best)]:.0%})' if best and rat[best] > 1.5 else 'not concentrated on one particle type'
    loc = 'local (same prong)' if e['same_prong'] > 0.7 else ('mostly self' if e['self'] > 0.5 else 'cross-prong')
    return f'{hlab(k)}: {q}, {loc}: same-prong {e["same_prong"]:.0%}, self {e["self"]:.0%}, mean query-key dR {e["dr"]:.2f}; {tgt}'


# ------------------------------------------------------------------------------------------------ main
def load_all():
    from interp.part_core import sub, cat, PRE, CL, sample
    rng = np.random.default_rng(SEED)
    ev, rf, rf2, sc = [], [], [], []
    for c in CL:
        n_sc = NSCAN if c == 'Tbqq' else 0
        J, rows = load_rows(TEST / f'{PRE[c]}_{FILE_NO}.root', NEVAL + 2 * NREF + n_sc, rng)
        perm = rng.permutation(len(rows))
        ev.append(sub(J, np.sort(perm[:NEVAL]))); rf.append(sub(J, np.sort(perm[NEVAL:NEVAL + NREF])))
        rf2.append(sub(J, np.sort(perm[NEVAL + NREF:NEVAL + 2 * NREF])))
        if n_sc: sc.append(sub(J, np.sort(perm[NEVAL + 2 * NREF:])))
        log(f'loaded {c}: {len(rows)} random rows of {PRE[c]}_{FILE_NO}.root (eval {NEVAL}, ref {NREF}, ref2 {NREF}, scan {n_sc})')
    B = rebuild(cat(ev)); BR = rebuild(cat(rf)); BR2 = rebuild(cat(rf2)); BSC = rebuild(cat(sc)); y = B['label'].astype(int)
    assert (BSC['label'].astype(int) == TOP).all()
    for k in range(10): assert (y == k).sum() == NEVAL
    sub_idx = np.sort(np.concatenate([rng.choice(np.where(y == k)[0], NSUB, replace=False) for k in range(10)]))
    P1 = rebuild(sample(1000, 3000))
    return rng, B, BR, BR2, BSC, y, sub_idx, P1


def main():
    import torch
    from interp.part_core import sub, kin, jmass, build
    OUTD.mkdir(parents=True, exist_ok=True); torch.manual_seed(SEED)
    rng, B, BR, BR2, BSC, y, sub_idx, P1 = load_all(); N = len(y); yP1 = P1['label'].astype(int)
    BSUB = sub(B, sub_idx); ysub = y[sub_idx]
    R = dict(node='P37', seed=SEED, smoke=SMOKE, files=f'test_20M <sample>_{FILE_NO}.root', NEVAL=NEVAL, NREF=NREF, NSCAN=NSCAN,
             NSUB=NSUB, NB=NB, NPERM=NPERM, NRAND=NRAND, NRJ=NRJ, KS=KS, KPATCH=KPATCH, heads=[hlab(x) for x in HEADS], classes=CLN,
             sample=dict(N_eval=N, N_ref=len(BR['label']), N_ref2=len(BR2['label']), N_scan=len(BSC['label']), N_sub=len(sub_idx), N_P1=len(yP1)))
    # ---------------- top-jet prongs and edits (deterministic, before any model)
    tj = np.where(y == TOP)[0]; BT = sub(B, tj); info = prongs(BT, rng)
    va = np.array([g is not None for g in info['gb']]); vb = np.array([g is not None for g in info['gw']]); vc = np.array([g is not None for g in info['gc']])
    vb2 = np.array([g is not None for g in info['gw2']])
    V = np.where(va & vb & vc)[0]; V2 = np.where(va & vb2)[0]; BV = sub(BT, V)
    Ea, mpa, _ = merge_edit(BV, [info['gb'][j] for j in V]); Eb, mpb, _ = merge_edit(BV, [info['gw'][j] for j in V])
    Ec, mpc, _ = merge_edit(BV, [info['gc'][j] for j in V])
    BV2 = sub(BT, V2); Eb2, _, _ = merge_edit(BV2, [info['gw2'][j] for j in V2]); Ea2, _, _ = merge_edit(BV2, [info['gb'][j] for j in V2])
    good = (info['mW'][V] > 65) & (info['mW'][V] < 95) & (info['mjet'][V] > 140) & (info['mjet'][V] < 205)
    dispb = info['disp_is_b'][V]; agree = info['ndisp_is_b'][V] == 1.0
    t32 = {e: np.array([tau32_of(p4_of(E_, j)) for j in range(len(V))]) for e, E_ in (('a', Ea), ('b', Eb), ('c', Ec))}
    t320 = info['tau32'][V]; Wv = boot_W(len(V), NB, rng)
    R['edit_checks'] = dict(
        n_top=int(len(tj)), n_valid=int(len(V)), n_valid_b2=int(len(V2)), frac_lt6=float(np.mean(BT['mask'].sum(1) < 6)),
        n_good_window=int(good.sum()), good_def='W-pair mass 65-95 GeV and jet mass 140-205 GeV',
        disp_is_b=dict(frac=float(np.nanmean(dispb)), n=int(np.isfinite(dispb).sum()), chance=1 / 3,
                       median_sigmax_b=float(np.nanmedian(info['sig_max_b'][V])), median_sigmax_w=float(np.nanmedian(info['sig_max_w'][V]))),
        ndisp_is_b=dict(frac=float(np.nanmean(info['ndisp_is_b'][V])), n=int(np.isfinite(info['ndisp_is_b'][V]).sum()),
                        mean_ndisp_b=float(np.nanmean(info['ndisp_b'][V])), mean_ndisp_w_max=float(np.nanmean(info['ndisp_w'][V])),
                        definition='subjet with the most charged tracks with |d0|/sigma_d0 > 3 and |d0| < 2 (JetClass units), ties excluded'),
        nb=dict(median=float(np.median(info['nb'][V])), mean=float(info['nb'][V].mean())),
        nw=dict(median=float(np.median(info['nw'][V])), mean=float(info['nw'][V].mean())),
        nc=dict(median=float(np.median(info['nc'][V])), mean=float(info['nc'][V].mean())),
        ptb_frac=float(np.median(info['ptb'][V] / BV['jet_pt'])), ptw_over_ptb=float(np.median(info['ptw'][V] / info['ptb'][V])),
        ptc_over_ptb=float(np.median(info['ptc'][V] / info['ptb'][V])),
        mjet_median=float(np.nanmedian(info['mjet'][V])), mW_median=float(np.nanmedian(info['mW'][V])),
        dmass_max={e: float(np.abs(jmass(E_) - jmass(BV)).max()) for e, E_ in (('a', Ea), ('b', Eb), ('c', Ec))},
        djetpt_max={e: float(np.abs(E_['jet_pt'] - BV['jet_pt']).max()) for e, E_ in (('a', Ea), ('b', Eb), ('c', Ec))},
        removed={e: float((BV['mask'].sum(1) - E_['mask'].sum(1)).mean()) for e, E_ in (('a', Ea), ('b', Eb), ('c', Ec))},
        tau32=dict(before=float(np.nanmedian(t320)), **{f'after_{e}': float(np.nanmedian(t32[e])) for e in t32},
                   **{f'd_{e}': mstat(np.nan_to_num(t32[e] - t320), Wv) for e in t32}))
    log('edit checks', json.dumps(strip(R['edit_checks'])))
    ex = next((q for q in range(len(V)) if good[q] and info['nb'][V[q]] >= 4), 0); jv = V[ex]; n0 = int(BV['mask'][ex].sum())
    pt0 = np.hypot(BV['px'][ex, :n0], BV['py'][ex, :n0]); na_ = int(Ea['mask'][ex].sum())
    R['example'] = dict(deta=BV['deta'][ex, :n0], dphi=BV['dphi'][ex, :n0], pt=pt0, lab=info['lab'][jv], mjet=info['mjet'][jv], mW=info['mW'][jv],
                        a_deta=Ea['deta'][ex, :na_], a_dphi=Ea['dphi'][ex, :na_], a_pt=np.hypot(Ea['px'][ex, :na_], Ea['py'][ex, :na_]),
                        a_merged_slot=int(np.min(info['gb'][jv])), mjet_a=float(jmass(sub(Ea, [ex]))[0]), jet_pt=float(BV['jet_pt'][ex]),
                        tau32_before=float(t320[ex]), tau32_a=float(t32['a'][ex]))
    # SCAN prong labels for the head description
    info_sc = prongs(BSC, np.random.default_rng(SEED + 5)); dsel = np.array([l is not None for l in info_sc['lab']])
    SF = np.where(np.random.default_rng(SEED + 1).random((NPERM, NSCAN)) < 0.5, -1.0, 1.0)
    SFv = np.where(np.random.default_rng(SEED + 2).random((NPERM, len(V))) < 0.5, -1.0, 1.0)
    Wsc = boot_W(NSCAN, NB, rng); Wall = boot_W(N, NB, rng); Wc = boot_W(NEVAL, NB, rng)
    cls_idx = [np.where(y == k)[0] for k in range(10)]
    for key in [k.strip() for k in ARGS.models.split(',') if k.strip()]:
        name, feat, null = MODEL_SPEC[key]; rngm = np.random.default_rng(SEED + 100 + list(MODEL_SPEC).index(key))
        log(f'==== {name}'); res = dict(name=name, null=null); M = HRun(feat, null=null); t1 = time.time()
        if null == 'perm': res['n_permuted_tensors'] = M.nperm
        res['acc_P1'] = acc_block(M.run(P1), yP1, conf=False)['acc10']
        L = M.run(B); res['eval'] = acc_block(L, y, Wall); corr0 = L.argmax(1) == y
        log(f'  acc P1 {res["acc_P1"]:.4f}  eval acc10 {res["eval"]["acc10"]:.4f} {np.round(res["eval"]["acc10_ci"], 4).tolist()} '
            f'top recall {res["eval"]["top_recall"]:.4f}  t/QCD acc {res["eval"]["bin_tq"]:.4f} AUC {res["eval"]["auc_tq"]:.4f} ({time.time() - t1:.0f}s)')
        M.run(BR2, dict(recmean=True)); Mh2, Mf2 = M.means(); M.reset_means()
        M.run(BR, dict(recmean=True)); Mh, Mf = M.means(); res['recon_max'] = M.recon
        res['mean_ablate_all_check'] = acc_block(M.run(BSUB, ctl_keep(M, [], Mh, Mf, range(10))), ysub, conf=False)
        # ---------------- step 1a: top scan
        t1 = time.time(); L0 = M.run(BSC); a0, b0 = ctr(L0, TOP, QCD), ctr(L0, TOP, WQQ)
        res['scan_unedited'] = dict(top_recall=float((L0.argmax(1) == TOP).mean()), lo_t=float(lodds(L0).mean()))
        dq, dw, dq2, lost = np.zeros((80, NSCAN)), np.zeros((80, NSCAN)), np.zeros((80, NSCAN)), np.zeros(80)
        trained = null is None
        for k in range(80):
            Lk = M.run(BSC, ctl_ablate(M, [k], Mh)); dq[k] = ctr(Lk, TOP, QCD) - a0; dw[k] = ctr(Lk, TOP, WQQ) - b0
            lost[k] = ((L0.argmax(1) == TOP) & (Lk.argmax(1) != TOP)).mean()
            if trained: dq2[k] = ctr(M.run(BSC, ctl_ablate(M, [k], Mh2)), TOP, QCD) - a0
        sq = [mstat(dq[k], Wsc, SF) for k in range(80)]; sw = [mstat(dw[k], Wsc, SF) for k in range(80)]
        padj = holm([s['p_perm'] for s in sq] + [s['p_perm'] for s in sw])
        scan = {}
        for k in range(80):
            scan[hlab(k)] = dict(d_QCD=sq[k], d_W=sw[k], p_holm_QCD=padj[k], p_holm_W=padj[80 + k], top_calls_lost=float(lost[k]))
            if trained: scan[hlab(k)]['d_QCD_ref2'] = float(dq2[k].mean())
        score = np.array([(sq[k]['v'] + sw[k]['v']) / 2 for k in range(80)])
        sig_neg = [k for k in range(80) if (padj[k] < 0.05 and sq[k]['v'] < 0) or (padj[80 + k] < 0.05 and sw[k]['v'] < 0)]
        sig_pos = [k for k in range(80) if (padj[k] < 0.05 and sq[k]['v'] > 0) or (padj[80 + k] < 0.05 and sw[k]['v'] > 0)]
        rank = sorted(sig_neg, key=lambda k: score[k]) + sorted([k for k in range(80) if k not in sig_neg], key=lambda k: score[k])
        mq = np.array([s['v'] for s in sq]); neg = np.clip(-mq, 0, None); negs = np.sort(neg)[::-1]
        res['scan'] = dict(heads=scan, n_tests=160, sig_neg=[hlab(k) for k in sig_neg], sig_pos=[hlab(k) for k in sig_pos],
                           n_sig_neg=len(sig_neg), n_sig_pos=len(sig_pos), rank=[hlab(k) for k in rank], _rank=rank,
                           score=[float(score[k]) for k in range(80)],
                           neg_share_top1=float(negs[0] / max(negs.sum(), 1e-12)), neg_share_top5=float(negs[:5].sum() / max(negs.sum(), 1e-12)),
                           neg_share_top10=float(negs[:10].sum() / max(negs.sum(), 1e-12)),
                           n_heads_for_50pct_neg=int(np.searchsorted(np.cumsum(negs) / max(negs.sum(), 1e-12), 0.5) + 1),
                           sum_single_dQCD=float(mq.sum()))
        if trained:
            dn = (dq - dq2).mean(1); res['scan']['noise_floor'] = dict(
                median_abs_mean_diff=float(np.median(np.abs(dn))), max_abs_mean_diff=float(np.abs(dn).max()),
                corr_ref_ref2=float(np.corrcoef(mq, dq2.mean(1))[0, 1]),
                sig_neg_same_sign_ref2=float(np.mean([dq2[k].mean() < 0 for k in sig_neg])) if sig_neg else None)
        log(f'  scan ({time.time() - t1:.0f}s): Holm-significant lowering heads {len(sig_neg)}, raising {len(sig_pos)}; top-10 share of '
            f'negative single effects {res["scan"]["neg_share_top10"]:.3f}; rank top 10 '
            + ', '.join(f'{hlab(k)} dQCD {sq[k]["v"]:+.3f} dW {sw[k]["v"]:+.3f}' for k in rank[:10]))
        # joint knock-outs on SCAN: top-k ranked together vs all 80, vs random k-sets
        t1 = time.time(); joint = {}
        Lall = M.run(BSC, ctl_ablate(M, range(80), Mh)); d80 = float((ctr(Lall) - a0).mean())
        for k in [q for q in KS if q > 0]:
            Lk = M.run(BSC, ctl_ablate(M, rank[:k], Mh)); dk = ctr(Lk) - a0
            rj = [float((ctr(M.run(BSC, ctl_ablate(M, rngm.choice(80, k, replace=False), Mh))) - a0).mean()) for _ in range(NRJ)] if k < 80 else []
            joint[str(k)] = dict(dQCD=mstat(dk, Wsc), share_of_all80=float(dk.mean() / d80) if d80 != 0 else None,
                                 share_ci=ci((Wsc @ dk) / (Wsc @ (ctr(Lall) - a0))), top_recall=float((Lk.argmax(1) == TOP).mean()),
                                 random=dict(mean=float(np.mean(rj)), min=float(np.min(rj)), max=float(np.max(rj))) if rj else None)
        res['scan']['joint'] = dict(d_all80=d80, by_k=joint)
        log(f'  joint ({time.time() - t1:.0f}s): all-80 dQCD {d80:+.3f}; ' + '; '.join(
            f'k={k} {joint[k]["dQCD"]["v"]:+.3f} share {joint[k]["share_of_all80"]:.2f}' + (f' (random {joint[k]["random"]["mean"]:+.3f})' if joint[k]['random'] else '') for k in joint))
        # ---------------- step 1b: 80 x 10 matrix on EVAL
        t1 = time.time(); lo0 = np.stack([lodds(L, k) for k in range(10)], 1)
        drec = np.zeros((80, 10)); drec_ci = np.zeros((80, 10, 2)); dlo = np.zeros((80, 10)); dlo_ci = np.zeros((80, 10, 2))
        pmc = np.ones((80, 10)); lostc = np.zeros((80, 10), int); gainc = np.zeros((80, 10), int); drec2 = np.full((80, 10), np.nan)
        for k in range(80):
            Lk = M.run(B, ctl_ablate(M, [k], Mh)); ck = Lk.argmax(1) == y
            for c in range(10):
                ix = cls_idx[c]; d = ck[ix].astype(float) - corr0[ix].astype(float); drec[k, c] = d.mean(); drec_ci[k, c] = ci(Wc @ d)
                dl = lodds(Lk[ix], c) - lo0[ix, c]; dlo[k, c] = dl.mean(); dlo_ci[k, c] = ci(Wc @ dl)
                lostc[k, c], gainc[k, c] = mcnemar_counts(corr0[ix], ck[ix]); pmc[k, c] = mcnemar_p(lostc[k, c], gainc[k, c])
            if key == 'full':
                ck2 = M.run(B, ctl_ablate(M, [k], Mh2)).argmax(1) == y
                for c in range(10): drec2[k, c] = (ck2[cls_idx[c]].astype(float) - corr0[cls_idx[c]]).mean()
            if k % 20 == 19: log(f'  matrix {k + 1}/80 ({time.time() - t1:.0f}s)')
        q = bh(pmc.ravel()).reshape(80, 10)
        sets_bh = {c: [k for k in range(80) if q[k, c] < 0.05 and drec[k, c] < 0] for c in range(10)}
        sets_top5 = {c: list(np.argsort(dlo[:, c])[:5]) for c in range(10)}
        jac = lambda a, b: (len(set(a) & set(b)) / len(set(a) | set(b))) if (set(a) | set(b)) else None
        groups = dict(top=[TOP], WZ=[WQQ, ZQQ], lepton=[TBL, HQQL])
        gs_bh = {g: sorted(set().union(*[sets_bh[c] for c in cs])) for g, cs in groups.items()}
        gs_5 = {g: sorted(set().union(*[sets_top5[c] for c in cs])) for g, cs in groups.items()}
        # null for fixed-size Jaccard: random sets of the same sizes
        rj5 = [jac(rngm.choice(80, len(gs_5['top']), replace=False), rngm.choice(80, len(gs_5['WZ']), replace=False)) for _ in range(2000)]
        mat = dict(drec=drec, drec_ci=drec_ci, dlo=dlo, dlo_ci=dlo_ci, p_mcnemar=pmc, q_bh=q, lost=lostc, gained=gainc,
                   n_bh_sig=int((q < 0.05).sum()), n_bh_sig_drop=int(((q < 0.05) & (drec < 0)).sum()),
                   sets_bh={CLN[c]: [hlab(k) for k in sets_bh[c]] for c in range(10)},
                   sets_top5_dlo={CLN[c]: [hlab(k) for k in sets_top5[c]] for c in range(10)},
                   group_sets_bh={g: [hlab(k) for k in v] for g, v in gs_bh.items()}, group_sets_top5={g: [hlab(k) for k in v] for g, v in gs_5.items()},
                   jaccard_bh={f'{a}-{b}': jac(gs_bh[a], gs_bh[b]) for a, b in (('top', 'WZ'), ('top', 'lepton'), ('WZ', 'lepton'))},
                   jaccard_top5={f'{a}-{b}': jac(gs_5[a], gs_5[b]) for a, b in (('top', 'WZ'), ('top', 'lepton'), ('WZ', 'lepton'))},
                   jaccard_top5_random=dict(mean=float(np.mean(rj5)), p95=float(np.percentile(rj5, 95))),
                   jaccard10_bh=[[jac(sets_bh[a], sets_bh[b]) for b in range(10)] for a in range(10)],
                   jaccard10_top5=[[jac(sets_top5[a], sets_top5[b]) for b in range(10)] for a in range(10)],
                   prior_heads={h: {CLN[c]: dict(drec=float(drec[HL.index(h), c]), q=float(q[HL.index(h), c]), dlo=float(dlo[HL.index(h), c]))
                                    for c in range(10)} for h in ('b2h1', 'b2h8', 'b1h1', 'b1h2', 'b1h3', 'b1h4', 'b1h5', 'b1h6', 'b1h7', 'b1h8')},
                   max_abs_drec=float(np.abs(drec).max()))
        if key == 'full':
            dd = drec - drec2; sig = (q < 0.05)
            mat['noise_floor_ref2'] = dict(median_abs_diff=float(np.median(np.abs(dd))), p95_abs_diff=float(np.percentile(np.abs(dd), 95)),
                                           max_abs_diff=float(np.abs(dd).max()), corr=float(np.corrcoef(drec.ravel(), drec2.ravel())[0, 1]),
                                           sig_same_sign=float(np.mean(np.sign(drec2[sig]) == np.sign(drec[sig]))) if sig.any() else None, drec_ref2=drec2)
        res['matrix'] = mat
        log(f'  matrix done ({time.time() - t1:.0f}s): BH-significant cells {mat["n_bh_sig"]} (drops {mat["n_bh_sig_drop"]}); max |drec| {mat["max_abs_drec"]:.3f}; '
            f'group sets BH {json.dumps(mat["group_sets_bh"])}; Jaccard BH {mat["jaccard_bh"]}; Jaccard top5 {mat["jaccard_top5"]} (random {mat["jaccard_top5_random"]})')
        fn = fn_for(key); fn.write_text(json.dumps(strip(dict(R, model=res)), indent=1))
        if not trained:
            res['runtime_s'] = time.time() - T0; fn.write_text(json.dumps(strip(dict(R, model=res)), indent=1)); log(f'  wrote {fn}')
            del M; torch.cuda.empty_cache(); continue
        # ---------------- step 4 (early, cheap): head description
        if key == 'full':
            t1 = time.time(); Bd = sub(BSC, np.where(dsel)[0]); labs = [info_sc['lab'][j] for j in np.where(dsel)[0]]
            res['describe'], res['describe_base'] = describe(M, Bd, labs)
            log(f'  describe ({time.time() - t1:.0f}s): ' + ' | '.join(res['describe'][hlab(k)]['text'] for k in rank[:6]))
        # ---------------- step 2: literal circuits
        t1 = time.time(); circ = {v_: {} for v_ in VARIANTS}
        rsets = {k: [rngm.choice(80, k, replace=False) for _ in range(NRAND)] for k in KS if 0 < k < 80}
        for k in KS:
            keep = rank[:k]
            for variant, mlp_idx in VARIANTS.items():
                ctl = ctl_keep(M, keep, Mh, Mf, mlp_idx)
                e = acc_block(M.run(B, ctl), y, Wall); es = acc_block(M.run(BSUB, ctl), ysub, conf=False)
                e['heads'] = [hlab(q_) for q_ in keep]; e['n_holm_sig_in'] = int(sum(q_ in sig_neg for q_ in keep)); e['sub'] = es
                if 0 < k < 80:
                    rr = dict(top_recall=[], balanced=[], bin_tq=[], acc10=[], auc_tq=[])
                    for ks_ in rsets[k]:
                        er = acc_block(M.run(BSUB, ctl_keep(M, ks_, Mh, Mf, mlp_idx)), ysub, conf=False)
                        for q_ in rr: rr[q_].append(er[q_] if er[q_] is not None else np.nan)
                    e['random'] = {q_: dict(mean=float(np.nanmean(v)), p95=float(np.nanpercentile(v, 95)), p5=float(np.nanpercentile(v, 5)),
                                            max=float(np.nanmax(v)), p_emp=float((1 + sum(x >= es[q_] for x in v)) / (1 + NRAND))) for q_, v in rr.items()}
                circ[variant][str(k)] = e
                log(f'  {variant} k={k}: top recall {e["top_recall"]:.3f} t/QCD acc {e["bin_tq"]:.3f} AUC {e["auc_tq"]:.3f} balanced {e["balanced"]:.3f} '
                    f'acc10 {e["acc10"]:.3f} distinct calls {e["n_distinct_calls"]}'
                    + (f' | random balanced {e["random"]["balanced"]["mean"]:.3f} (p95 {e["random"]["balanced"]["p95"]:.3f}) p {e["random"]["balanced"]["p_emp"]:.3f}; '
                       f'AUC {e["random"]["auc_tq"]["mean"]:.3f} p {e["random"]["auc_tq"]["p_emp"]:.3f}' if 'random' in e else '') + f' ({time.time() - t1:.0f}s)')
        res['circuits'] = circ; fn.write_text(json.dumps(strip(dict(R, model=res)), indent=1))
        # ---------------- step 3: edits and patching
        t1 = time.time(); sets = {}
        for k in KPATCH:
            sets[f'circ{k}'] = rank[:k]
            for r_ in range(NRAND): sets[f'r{k}_{r_}'] = rngm.choice(80, k, replace=False).tolist()
        ed = {}
        for en, E_, mp_ in (('a', Ea, mpa), ('b', Eb, mpb), ('c', Ec, mpc)):
            L0e, L1e, LP = patch_runs(M, BV, E_, mp_, sets)
            dfull = lodds(L1e) - lodds(L0e); dq_full = ctr(L1e) - ctr(L0e)
            e = dict(dlo_t=mstat(dfull, Wv, SFv), dQCD=mstat(dq_full, Wv, SFv), calls_changed=float((L0e.argmax(1) != L1e.argmax(1)).mean()),
                     mcnemar_top=mcnemar(L0e.argmax(1) == TOP, L1e.argmax(1) == TOP), changed_to=np.bincount(L1e.argmax(1)[L0e.argmax(1) != L1e.argmax(1)], minlength=10),
                     dlo_t_good=float(dfull[good].mean()) if good.any() else None,
                     dlo_t_disp_agree=mstat(dfull[agree], boot_W(int(agree.sum()), NB, rng)) if agree.sum() > 2 else None, _d=dfull)
            for k in KPATCH:
                dp = lodds(LP[f'circ{k}']) - lodds(L0e); shb = (Wv @ dp) / (Wv @ dfull)
                sh = float(dp.mean() / dfull.mean()); rs = [float((lodds(LP[f'r{k}_{r_}']) - lodds(L0e)).mean() / dfull.mean()) for r_ in range(NRAND)]
                e[f'patch{k}'] = dict(heads=[hlab(q_) for q_ in sets[f'circ{k}']], dpatch=mstat(dp, Wv), share=sh, share_ci=ci(shb),
                                      random_share=dict(mean=float(np.mean(rs)), p95=float(np.percentile(rs, 95)), p5=float(np.percentile(rs, 5)),
                                                        max=float(np.max(rs)), p_emp=float((1 + sum(x >= sh for x in rs)) / (1 + NRAND))))
            ed[en] = e
            log(f'  edit {en}: d lo_t {e["dlo_t"]["v"]:+.3f} {np.round(e["dlo_t"]["ci"], 3).tolist()} dQCD {e["dQCD"]["v"]:+.3f} calls changed {e["calls_changed"]:.3f} '
                f'top {e["mcnemar_top"]["before"]:.3f}->{e["mcnemar_top"]["after"]:.3f} | '
                + '; '.join(f'circ{k} share {e[f"patch{k}"]["share"]:.3f} {np.round(e[f"patch{k}"]["share_ci"], 3).tolist()} vs random {e[f"patch{k}"]["random_share"]["mean"]:.3f} '
                            f'(p95 {e[f"patch{k}"]["random_share"]["p95"]:.3f}) p {e[f"patch{k}"]["random_share"]["p_emp"]:.3f}' for k in KPATCH) + f' ({time.time() - t1:.0f}s)')
        ed['a_vs_b'] = dict(ddlo_t=mstat(ed['a']['_d'] - ed['b']['_d'], Wv, SFv)); ed['a_vs_c'] = dict(ddlo_t=mstat(ed['a']['_d'] - ed['c']['_d'], Wv, SFv))
        L0c = M.run(BV2); Lb2 = M.run(Eb2); La2 = M.run(Ea2); d2 = lodds(Lb2) - lodds(L0c); da2 = lodds(La2) - lodds(L0c)
        W2 = boot_W(len(V2), NB, rng); SF2 = np.where(np.random.default_rng(SEED + 3).random((NPERM, len(V2))) < 0.5, -1.0, 1.0)
        ed['b2'] = dict(n=int(len(V2)), dlo_t=mstat(d2, W2, SF2), a_on_same=mstat(da2, W2), ddlo_t_a_minus_b2=mstat(da2 - d2, W2, SF2),
                        calls_changed=float((L0c.argmax(1) != Lb2.argmax(1)).mean()), mcnemar_top=mcnemar(L0c.argmax(1) == TOP, Lb2.argmax(1) == TOP))
        Ls = M.run(build(Ea, *kin(Ea))); Lu = M.run(Ea); ed['layout_check_max_abs_dlogit'] = float(np.abs(Ls - Lu).max()); ed['layout_check_median_abs_dlogit'] = float(np.median(np.abs(Ls - Lu)))
        log(f'  a - b: {ed["a_vs_b"]["ddlo_t"]["v"]:+.3f} {np.round(ed["a_vs_b"]["ddlo_t"]["ci"], 3).tolist()} p {ed["a_vs_b"]["ddlo_t"]["p_perm"]:.2g}; '
            f'a - c: {ed["a_vs_c"]["ddlo_t"]["v"]:+.3f} p {ed["a_vs_c"]["ddlo_t"]["p_perm"]:.2g}; b2 {ed["b2"]["dlo_t"]["v"]:+.3f} (a on same {ed["b2"]["a_on_same"]["v"]:+.3f}); '
            f'layout check {ed["layout_check_max_abs_dlogit"]:.2e}')
        res['edits'] = ed; res['runtime_s'] = time.time() - T0
        fn.write_text(json.dumps(strip(dict(R, model=res)), indent=1)); log(f'  wrote {fn}')
        del M; torch.cuda.empty_cache()
    log('done')


HL = [hlab(k) for k in range(80)]


# ------------------------------------------------------------------------------------------------ figure
def plot():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    D = {k: json.loads(fn_for(k).read_text()) for k in MODEL_SPEC if fn_for(k).exists()}
    R = D['full']; F = R['model']
    fig = plt.figure(figsize=(17, 10.5)); gs = fig.add_gridspec(2, 3)
    # A: example jet before/after
    ex = R['example']; axA = fig.add_subplot(gs[0, 0]); cols = np.array(['#c55a11', '#1f4e79', '#4f81bd'])
    lab = np.asarray(ex['lab']); pt = np.asarray(ex['pt']); apt = np.asarray(ex['a_pt']); i0 = ex['a_merged_slot']
    axA.scatter(ex['deta'], ex['dphi'], s=8 + 400 * pt / pt.max(), c=cols[lab], alpha=0.6, edgecolors='none')
    axA.scatter([ex['a_deta'][i0]], [ex['a_dphi'][i0]], s=8 + 400 * apt[i0] / pt.max(), facecolors='none', edgecolors='k', lw=1.5)
    axA.set_xlabel(r'$\Delta\eta$ to jet axis'); axA.set_ylabel(r'$\Delta\phi$ to jet axis [rad]'); axA.set_aspect('equal')
    axA.set_title(f"Edit on a real t->bqq' jet (pT {ex['jet_pt']:.0f} GeV, m {ex['mjet']:.0f} GeV): the {int((lab == 0).sum())} b-prong particles\n"
                  f"(orange) become one particle (black ring); jet mass after {ex['mjet_a']:.0f} GeV", fontsize=9)
    axA.legend(handles=[Line2D([], [], marker='o', ls='', color='#c55a11', label='b-prong (3rd kT subjet)'),
                        Line2D([], [], marker='o', ls='', color='#1f4e79', label='W prongs (pair mass nearest 80 GeV)'),
                        Line2D([], [], marker='o', ls='', mfc='none', mec='k', label='merged b-prong after the edit')], fontsize=7, loc='lower left')
    # B: 80 x 10 recall-change matrix
    axB = fig.add_subplot(gs[0, 1:]); v = np.array(F['matrix']['drec']).T; q = np.array(F['matrix']['q_bh']).T
    lim = max(np.abs(v).max(), 0.02); im = axB.imshow(v, cmap='RdBu', vmin=-lim, vmax=lim, aspect='auto', interpolation='nearest')
    for c in range(10):
        for k in range(80):
            if q[c, k] < 0.05 and v[c, k] < 0: axB.text(k, c, '*', ha='center', va='center', fontsize=6)
    axB.set_yticks(range(10)); axB.set_yticklabels(R['classes'], fontsize=8)
    axB.set_xticks([4 + 8 * i for i in range(10)]); axB.set_xticklabels([f'particle\nblock {i + 1}' for i in range(8)] + ['class\nblock 1', 'class\nblock 2'], fontsize=7)
    for i in range(1, 10): axB.axvline(8 * i - 0.5, color='k', lw=0.4)
    plt.colorbar(im, ax=axB, fraction=0.02).set_label('change in recall of the class', fontsize=8)
    axB.set_title(f"ParT_full: one of 80 heads replaced by its mean output at a time (1,000 jets per class); colour = change in the share of\n"
                  f"each class's jets called correctly; * = drop significant after Benjamini-Hochberg over 800 tests (McNemar)", fontsize=9)
    # C: joint knock-out share
    axC = fig.add_subplot(gs[1, 0]); ks = [k for k in R['KS'] if 0 < k < 80]
    for key, col in (('full', '#1f4e79'), ('kin', '#c55a11')):
        if key not in D: continue
        J = D[key]['model']['scan']['joint']; d80 = J['d_all80']
        axC.plot(ks, [J['by_k'][str(k)]['dQCD']['v'] / d80 for k in ks], 'o-', color=col, label=f'{D[key]["model"]["name"]}: top-ranked k heads')
        axC.plot(ks, [J['by_k'][str(k)]['random']['mean'] / d80 for k in ks], 'x--', color=col, alpha=0.6, label=f'{D[key]["model"]["name"]}: random k heads (mean of {R["NRJ"]})')
    axC.axhline(0.5, color='k', ls=':', lw=1); axC.set_xscale('log'); axC.set_xticks(ks); axC.set_xticklabels(ks)
    axC.set_xlabel('heads replaced by their mean together (k)'); axC.set_ylabel('share of the drop from replacing all 80')
    axC.set_title("True t->bqq' jets: drop in log(p_t / p_QCD) when k heads are\nreplaced together, relative to replacing all 80 heads", fontsize=9); axC.legend(fontsize=7)
    # D: literal circuits balanced accuracy
    axD = fig.add_subplot(gs[1, 1])
    for var, col, lab_ in (('literal', '#1f4e79', 'only k heads kept (all MLPs at mean)'), ('cls_mlp', '#c55a11', 'k heads + class-block MLPs kept'),
                           ('heads_only', '#7f7f7f', 'k heads + all MLPs kept')):
        cc = F['circuits'][var]
        axD.plot(ks, [cc[str(k)]['sub']['balanced'] for k in ks], 'o-', color=col, label=lab_)
        axD.fill_between(ks, [cc[str(k)]['random']['balanced']['p5'] for k in ks], [cc[str(k)]['random']['balanced']['p95'] for k in ks], color=col, alpha=0.15)
    axD.axhline(F['eval']['balanced'], color='k', lw=1, label='full network'); axD.axhline(0.1, color='k', ls=':', lw=1)
    axD.set_xscale('log'); axD.set_xticks(ks); axD.set_xticklabels(ks); axD.set_ylim(0, 1); axD.set_xlabel('heads kept (k), ranked by the top scan')
    axD.set_ylabel('10-class balanced accuracy'); axD.legend(fontsize=7)
    axD.set_title(f'ParT_full: circuits kept alone (lines) vs {R["NRAND"]} random sets of k heads\n(bands: 5-95 % of random sets)', fontsize=9)
    # E: edits
    axE = fig.add_subplot(gs[1, 2]); groups = []
    for key in ('full', 'kin'):
        if key not in D: continue
        e = D[key]['model']['edits']
        for en, nm in (('a', 'b-prong'), ('b', 'W-prong\npT-matched'), ('c', 'one W\nsubjet')):
            groups.append((f'{D[key]["model"]["name"].replace("ParT_", "")}\n{nm}', e[en]))
    x = np.arange(len(groups)); w = 0.27
    full = [g[1]['dlo_t']['v'] for g in groups]; err = np.array([[g[1]['dlo_t']['v'] - g[1]['dlo_t']['ci'][0], g[1]['dlo_t']['ci'][1] - g[1]['dlo_t']['v']] for g in groups]).T
    axE.bar(x - w / 2, full, w, yerr=err, color='#7f7f7f', label='whole network sees the edit')
    axE.bar(x + w / 2, [g[1]['patch5']['dpatch']['v'] for g in groups], w, color='#1f4e79', label='only the top-5 heads see the edit')
    axE.plot(x + w / 2, [g[1]['patch5']['random_share']['mean'] * g[1]['dlo_t']['v'] for g in groups], '_', ms=14, mew=2, color='#c55a11',
             label=f'mean of {R["NRAND"]} random 5-head sets')
    axE.set_xticks(x); axE.set_xticklabels([g[0] for g in groups], fontsize=7); axE.axhline(0, color='k', lw=0.5)
    axE.set_ylabel(r"mean change in $\log[p_t/(1-p_t)]$"); axE.legend(fontsize=7)
    axE.set_title(f"Merging one prong into a single particle at fixed jet\nfour-momentum (N = {R['edit_checks']['n_valid']} true t->bqq' jets)", fontsize=9)
    fig.tight_layout(); p = OUTD / f'p37_top_circuit{TAG}.png'; fig.savefig(p, dpi=110); print('wrote', p)


if __name__ == '__main__':
    plot() if ARGS.plot else main()
