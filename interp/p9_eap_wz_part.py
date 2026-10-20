"""P9 — whole-network W-vs-Z circuit discovery in ParT_full: edge attribution patching (EAP-IG over edge masks) + ACDC-style
greedy pruning, faithfulness vs circuit size, literal-circuit accuracy and 10-class accuracy/confusion (claim: circuit
completeness; Paper C weakness 1).

Graph (exact decomposition of weaver-core 0.4.17's ParticleTransformer, checked against the model's logits):
  writer nodes (433):
    E           the particle embedding (model.embed output; the input of block 1)
    P{k}B{h}    pair bias of particle block k, head h (model.pair_embed output channel h, added to head h's attention logits)
    P{k}H{h}    particle block k, head h: y = R(ctx_h W_o,h^T), R = weaver's c_attn per-'head' scale + 'tbhd->tbdh' reshuffle
    P{k}M{g}    particle block k, MLP group g: 32 hidden units (post_fc_norm outputs) times their fc2 columns; groups are formed
                per block by ranking the 512 units by |attribution| (top 32 = group 0, ...), from a unit-level attribution pre-pass
    C{b}H{h}    class-attention block b, head h;  C{b}M{g} class block b MLP group g (same grouping rule)
    z{i}        coordinate i of the class token after the final LayerNorm (n98 = z97); logits = z W + b
  readers: every head (its own copy of the residual stream -> own pre_attn_norm -> q, k, v of that head only), every MLP group
    (own copy -> pre_fc_norm -> its 32 fc1 rows; GELU; post_fc_norm over all 512 units), class heads (own class-stream copy as
    query and own copy of the final particle stream as keys/values), the final LayerNorm (OUT), the logits (LOGIT, reads z).
  An edge u -> v carries u's contribution to v's input. post_attn_norm mixes the heads of a block, so v sees
  LN(sum_h [m_uv y_h + (1 - m_uv) y_h^corr] + bias) for each earlier block; MLP and embedding contributions are linear and are
  scaled by the w_resid factors between writer and reader (exact). 21,504 edges.
  Edge masks m in [0, 1]: m = 1 everywhere reproduces the model; m = 0 everywhere gives the donor jet's logits.
  Ablation = resample: an ablated edge carries the donor jet's own (clean) value of u (slot-wise for particle nodes; the clean
  jet's padding mask is used).

Samples (JetClass validation split /j-jepa-vol/JetClass/Pythia/val_5M; random rows, never first-k):
  discovery: NPER random rows each of WToQQ_123.root and ZToQQ_123.root, numpy default_rng(2026)
  evaluation: NPER random rows each of the same files, disjoint from discovery, default_rng(2027); eval[::5] for controls
  all-class: NALL random rows of each class's <sample>_124.root, default_rng(2028) -> 10-class read
  P1 sample (test files, rows 3,000-4,000 per class) -> unedited accuracy check only.
Corruptions (donors):
  B (main): W <-> Z donor with matched jet pT (one of the 5 nearest in pT among the other class of the same set)
  A: same class, |m_donor - m| < 5 GeV, nearest multiplicity among those (random among the 3 nearest)
  U (10-class set): uniform random other jet of the all-class set (P3's uniform_S convention)
Metric: d = l_W - l_Z; s = +1 (true W), -1 (true Z). Faithfulness F = sum s (d_circ - d_corr) / sum s (d_full - d_corr).
Step 0: unit attribution of every MLP hidden unit (1,000 discovery jets, B donors): sum (h_corr - h) dD/dh -> grouping.
Step 1: EAP-IG over edge masks: score_e = mean over alpha in {0.1, 0.3, 0.5, 0.7, 0.9} of dD/dm_e at m = alpha
        (D = sum_jets s d), on all discovery jets; also plain EAP (alpha = 1). Completeness check sum_e score_e = D(1) - D(0).
Step 2: ACDC-style greedy pruning, starting from the top ACDC_TOP edges by |score| (all others ablated), on NACDC discovery
        jets with cached donors: edges in reverse topological order (reader stage descending, |score| ascending); remove an
        edge if F drops by less than tau; nested passes tau = 0.001, 0.003, 0.01.
Outputs on the evaluation set: F and W/Z accuracy vs number of kept edges (top-k by |score|), vs number of kept nodes (top-k
nodes by summed outgoing score, node-induced = the user's literal circuit: every edge between kept nodes kept, everything else
resampled); the 3 ACDC circuits as edge circuits and as node-induced literal circuits: W/Z accuracy, F, McNemar vs the full
model, permutation p vs NRAND random circuits of matched size and category distribution (edges: per (writer type+block,
reader type+block); nodes: per type+block), corruption A, and the 10-class accuracy/confusion with U donors; reference circuits
{b2h8, b1h3, n98} and 16 class heads + n98, literal (whole network) and in P3's scope (particle stage intact); attribution
fidelity: single-edge ablation of the top NFID edges and NFID random edges vs their EAP-IG / EAP predictions (Spearman,
bootstrap, Holm). Bootstrap over jets (NB), 95 % intervals.

  ./part_job.sh c-p9-smoke 'python -m interp.p9_eap_wz_part --smoke'  -> results/part/p9_eap_wz_smoke.json
  ./part_job.sh c-p9-run 'python -m interp.p9_eap_wz_part'            -> results/part/p9_eap_wz.json, p9_eap_scores.npz, p9_eap_wz.png
  python -m interp.p9_eap_wz_part --plot                                (figure from the json)
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np

OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
VAL = Path(os.environ.get('P9_VAL', '/j-jepa-vol/JetClass/Pythia/val_5M'))
SMOKE = '--smoke' in sys.argv
E_ = lambda k, d: int(os.environ.get(k, d))
NPER = E_('P9_NPER', 120 if SMOKE else 5000)
NALL = E_('P9_NALL', 30 if SMOKE else 2000)
NUNIT = 100 if SMOKE else 1000
NB = 200 if SMOKE else 1000
NRAND = E_('P9_NRAND', 3 if SMOKE else 100)
NACDC = E_('P9_NACDC', 64 if SMOKE else 300)
ACDC_TOP = E_('P9_ACDC_TOP', 150 if SMOKE else 2000)
NFID = 5 if SMOKE else 50
BATCH = E_('P9_BATCH', 48)
IGB = E_('P9_IGB', 12)                       # gradient passes: at most IGB * 64 particle slots per batch
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:256')
TAUS = [0.001, 0.003, 0.01]
ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]
H, HD, C, NG, GS = 8, 16, 128, 16, 32
FN = OUT / ('p9_eap_wz_smoke.json' if SMOKE else 'p9_eap_wz.json')
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:6.0f} s]', *a, flush=True)


# ------------------------------------------------------------------------------------------------------------ statistics
def boot_W(n, nb, rng):
    W = np.zeros((nb, n), np.float32)
    for b in range(nb): W[b] = np.bincount(rng.integers(0, n, n), minlength=n)
    return W


def ci(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [None, None]


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.zeros(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj.tolist()


def mcnemar(ok0, ok1):
    from scipy.stats import binomtest
    b = int((ok0 & ~ok1).sum()); c = int((~ok0 & ok1).sum())
    return dict(right_to_wrong=b, wrong_to_right=c, p=(float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0))


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items() if not str(k).startswith('_')}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.ndarray): return clean(o.tolist())
    if isinstance(o, (np.floating,)): return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, float) and not math.isfinite(o): return None
    return o


# ------------------------------------------------------------------------------------------------------------ data
def load_sel(path, rowsets, chunk=25000):
    from interp.part_core import read_root, sub, cat
    allr = np.unique(np.concatenate(rowsets)); N = max(allr) + 1; parts = []
    import uproot
    N = uproot.open(path)['tree'].num_entries
    for s in range(0, N, chunk):
        r = allr[(allr >= s) & (allr < s + chunk)]
        if len(r): parts.append(sub(read_root(path, s, min(s + chunk, N)), r - s))
    J = cat(parts); pos = {int(r): i for i, r in enumerate(allr)}
    return [sub(J, np.array([pos[int(r)] for r in rs])) for rs in rowsets], N


def donors_B(pt, y, iW, iZ, rng):
    don = np.zeros(len(y), int)
    for a, b in ((iW, iZ), (iZ, iW)):
        ia, ib = np.where(y == a)[0], np.where(y == b)[0]; o = ib[np.argsort(pt[ib])]; ps = pt[o]
        for i in ia:
            j = np.searchsorted(ps, pt[i]); lo, hi = max(0, j - 5), min(len(o), j + 5)
            cand = o[lo:hi]; cand = cand[np.argsort(np.abs(pt[cand] - pt[i]))[:5]]; don[i] = rng.choice(cand)
    return don


def donors_A(mass, mult, y, rng):
    don = np.zeros(len(y), int); nfb = 0
    for c in np.unique(y):
        ix = np.where(y == c)[0]
        for i in ix:
            cand = ix[(np.abs(mass[ix] - mass[i]) < 5.0) & (ix != i)]
            if len(cand) == 0: cand = ix[ix != i][np.argsort(np.abs(mass[ix[ix != i]] - mass[i]))[:3]]; nfb += 1
            cand = cand[np.argsort(np.abs(mult[cand] - mult[i]), kind='stable')[:3]]; don[i] = rng.choice(cand)
    return don, nfb


def donors_U(n, rng):
    d = rng.integers(0, n - 1, n); return d + (d >= np.arange(n))


# ------------------------------------------------------------------------------------------------------------ graph
class Graph:
    def __init__(self):
        self.edges, self.ix = [], {}
        for k in range(8):
            rH = [f'P{k}H{h}' for h in range(H)]; rM = [f'P{k}M{g}' for g in range(NG)]
            self.add(f'P{k}lin', ['E'] + [f'P{j}M{g}' for j in range(k) for g in range(NG)], rH + rM)
            for j in range(k): self.add(f'P{k}att{j}', [f'P{j}H{h}' for h in range(H)], rH + rM)
            self.add(f'P{k}own', [f'P{k}H{h}' for h in range(H)], rM)
            self.ix[f'P{k}pair'] = np.array([[self._e(f'P{k}B{h}', f'P{k}H{h}')] for h in range(H)])
        for b in range(2):
            rH = [f'C{b}H{h}' for h in range(H)]; rM = [f'C{b}M{g}' for g in range(NG)]
            self.add(f'C{b}Xlin', ['E'] + [f'P{j}M{g}' for j in range(8) for g in range(NG)], rH)
            for j in range(8): self.add(f'C{b}Xatt{j}', [f'P{j}H{h}' for h in range(H)], rH)
            if b == 1:
                self.add('C1lin', [f'C0M{g}' for g in range(NG)], rH + rM); self.add('C1att0', [f'C0H{h}' for h in range(H)], rH + rM)
            self.add(f'C{b}own', [f'C{b}H{h}' for h in range(H)], rM)
        self.add('Flin', [f'C{b}M{g}' for b in range(2) for g in range(NG)], ['OUT'])
        self.add('Fatt0', [f'C0H{h}' for h in range(H)], ['OUT']); self.add('Fatt1', [f'C1H{h}' for h in range(H)], ['OUT'])
        self.add('z', [f'z{i}' for i in range(C)], ['LOGIT'])
        self.NE = len(self.edges); self.W = np.array([w for w, _ in self.edges]); self.R = np.array([r for _, r in self.edges])
        self.nodes = sorted(set(self.W.tolist()), key=self.node_order)
        self.rstage = np.array([self.stage(r) for r in self.R])
        self.ecat = np.array([f'{self.ncat(w)}>{self.ncat(r)}' for w, r in self.edges])

    def _e(self, w, r):
        self.edges.append((w, r)); return len(self.edges) - 1

    def add(self, key, rows, cols):
        self.ix[key] = np.array([[self._e(w, r) for r in cols] for w in rows])

    @staticmethod
    def stage(r):
        if r in ('OUT', 'LOGIT'): return 10
        return int(r[1]) if r[0] == 'P' else 8 + int(r[1])

    @staticmethod
    def ncat(n):
        if n in ('E', 'OUT', 'LOGIT'): return n
        if n[0] == 'z': return 'z'
        return n[:3]                       # P3H, P3M, P3B, C0H, C1M

    @staticmethod
    def node_order(n):
        if n == 'E': return (0, 0, 0, 0)
        if n[0] == 'z': return (3, 0, 0, int(n[1:]))
        t = {'B': 0, 'H': 1, 'M': 2}[n[2]]
        return (1 if n[0] == 'P' else 2, int(n[1]), t, int(n[3:]))

    def node_mask(self, nodes):
        S = set(nodes) | {'OUT', 'LOGIT'}
        return np.array([(w in S) and (r in S) for w, r in self.edges], np.float32)


# ------------------------------------------------------------------------------------------------------------ model
class Net:
    def __init__(self, G):
        import torch, torch.nn.functional as F
        from interp.part_core import ParTNetwork
        self.t, self.F, self.G = torch, F, G
        self.net = ParTNetwork('full'); self.m = self.net.model; self.m.eval(); dev = self.dev = self.net.device
        g = lambda p: p.detach().float().to(dev)
        ln = lambda l: (g(l.weight), g(l.bias), l.eps)
        hh, dd = np.arange(H), np.arange(HD)
        new = (dd[:, None] * H + hh[None, :]).ravel(); old = (HD * hh[None, :] + dd[:, None]).ravel()
        hs = np.broadcast_to(hh[None, :], (HD, H)).ravel()
        po = np.zeros(C, int); po[new] = old; hsel = np.zeros(C, int); hsel[new] = hs
        po, hsel = torch.from_numpy(po).to(dev), torch.from_numpy(hsel).to(dev)

        def blk(b):
            a = b.attn; Wi, bi = g(a.in_proj_weight), g(a.in_proj_bias); s = g(b.c_attn)[hsel]
            WoR = g(a.out_proj.weight)[po] * s[:, None]
            return dict(ln_pre=ln(b.pre_attn_norm), Wq=Wi[:C].view(H, HD, C), bq=bi[:C].view(H, HD), Wk=Wi[C:2 * C].view(H, HD, C),
                        bk=bi[C:2 * C].view(H, HD), Wv=Wi[2 * C:].view(H, HD, C), bv=bi[2 * C:].view(H, HD),
                        Wo=WoR.view(C, H, HD).permute(1, 0, 2).contiguous(), rb=g(a.out_proj.bias)[po] * s,
                        ln_pa=ln(b.post_attn_norm), ln_pf=ln(b.pre_fc_norm), W1=g(b.fc1.weight), b1=g(b.fc1.bias),
                        ln_po=ln(b.post_fc_norm), W2=g(b.fc2.weight), b2=g(b.fc2.bias), w=g(b.w_resid))
        self.PB = [blk(b) for b in self.m.blocks]; self.CB = [blk(b) for b in self.m.cls_blocks]
        self.cls0 = g(self.m.cls_token).reshape(C); self.lnf = ln(self.m.norm); self.Wfc, self.bfc = g(self.m.fc[0].weight), g(self.m.fc[0].bias)
        self.set_perm([np.arange(4 * C)] * 10)
        w = [B['w'] for B in self.PB]

        def sc(a, b):
            o = torch.ones(C, device=dev)
            for l in range(a, b): o = o * w[l]
            return o
        self.sE = [sc(0, k) for k in range(9)]; self.sA = {(j, k): sc(j, k) for k in range(9) for j in range(k)}
        self.sM = {(j, k): sc(j + 1, k) for k in range(9) for j in range(k)}
        self.cP = [sum((self.PB[j]['b2'] * self.sM[(j, k)] for j in range(k)), torch.zeros(C, device=dev)) for k in range(9)]
        self.ixt = {k: torch.from_numpy(v).to(dev) for k, v in G.ix.items()}

    def set_perm(self, perms):
        t = self.t
        for B, p in zip(self.PB + self.CB, perms):
            pt = t.from_numpy(np.asarray(p)).to(self.dev); B['perm'] = np.asarray(p)
            B['W1g'] = B['W1'][pt].view(NG, GS, C); B['b1g'] = B['b1'][pt].view(NG, GS)
            B['ln_pop'] = (B['ln_po'][0][pt], B['ln_po'][1][pt], B['ln_po'][2])
            B['W2g'] = B['W2'][:, pt].view(C, NG, GS).permute(1, 0, 2).contiguous()

    # ---- pieces
    def ln(self, x, p):
        return self.F.layer_norm(x, (x.shape[-1],), p[0], p[1], p[2])

    def lnA(self, B, y):
        return self.ln(y, B['ln_pa'])

    def qkv(self, u, W, b):
        """u (V, n, ..., C) with V in {1, H} -> (H, n, ..., HD)"""
        t = self.t
        o = t.einsum('n...c,hdc->hn...d', u[0], W) if u.shape[0] == 1 else t.einsum('hn...c,hdc->hn...d', u, W)
        return o + b.view(H, *([1] * (o.dim() - 2)), HD)

    def pattn(self, B, xh, Ub, pad):
        t = self.t; u = self.ln(xh, B['ln_pre'])
        q = self.qkv(u, B['Wq'], B['bq']) / math.sqrt(HD); k = self.qkv(u, B['Wk'], B['bk']); v = self.qkv(u, B['Wv'], B['bv'])
        a = t.einsum('hnpd,hnqd->hnpq', q, k) + Ub.permute(1, 0, 2, 3)
        a = a.masked_fill(pad[None, :, None, :], float('-inf')).softmax(-1)
        return t.einsum('hnpd,hcd->hnpc', t.einsum('hnpq,hnqd->hnpd', a, v), B['Wo'])

    def cattn(self, B, rh, Xb, padc):
        t = self.t; V = max(rh.shape[0], Xb.shape[0])
        if V > 1: rh, Xb = rh.expand(V, *rh.shape[1:]), Xb.expand(V, *Xb.shape[1:])
        u = self.ln(t.cat([rh[:, :, None, :], Xb], 2), B['ln_pre'])
        k = self.qkv(u, B['Wk'], B['bk']); v = self.qkv(u, B['Wv'], B['bv']); q = self.qkv(rh, B['Wq'], B['bq']) / math.sqrt(HD)
        a = t.einsum('hnd,hnpd->hnp', q, k).masked_fill(padc[None], float('-inf')).softmax(-1)
        return t.einsum('hnd,hcd->hnc', t.einsum('hnp,hnpd->hnd', a, v), B['Wo'])

    def mlp(self, B, xm):
        """xm (V, n, ..., C), V in {1, NG} -> h (n, ..., 512) in B['perm'] order"""
        t = self.t; u = self.ln(xm, B['ln_pf'])
        a = t.einsum('n...c,guc->gn...u', u[0], B['W1g']) if u.shape[0] == 1 else t.einsum('gn...c,guc->gn...u', u, B['W1g'])
        a = self.F.gelu(a + B['b1g'].view(NG, *([1] * (a.dim() - 2)), GS))
        h = a.movedim(0, -2).reshape(*a.shape[1:-1], NG * GS)
        return self.ln(h, B['ln_pop'])

    def contrib(self, B, h):
        return self.t.einsum('n...gu,gcu->gn...c', h.view(*h.shape[:-1], NG, GS), B['W2g'])

    def pstream(self, k, st, E, cz, M, klin, katt):
        t = self.t
        if M is None:
            x = E * self.sE[k] + self.cP[k]
            for j in range(k):
                x = x + st['mg'][j].sum(0) * self.sM[(j, k)] + self.lnA(self.PB[j], self.PB[j]['rb'] + st['y'][j].sum(0)) * self.sA[(j, k)]
            return x[None]
        key = ('lin', k)
        if key not in st:
            cur = t.cat([E[None]] + [st['mg'][j] for j in range(k)]); cor = t.cat([cz['E'][None]] + [cz['mg'][j] for j in range(k)])
            sc = t.stack([self.sE[k]] + [self.sM[(j, k)] for j in range(k) for _ in range(NG)])[:, None, None, :]
            st[key] = (((cor * sc).sum(0) + self.cP[k])[None], (cur - cor) * sc); del cur, cor
        base, dl = st[key]
        x = base + t.einsum('uv,unpc->vnpc', M(klin), dl)
        if k < 8: del st[key]
        for j in range(k):
            yc = cz['y'][j]
            Yv = (self.PB[j]['rb'] + yc.sum(0))[None] + t.einsum('hv,hnpc->vnpc', M(katt.format(j)), st['y'][j] - yc)
            x = x + self.lnA(self.PB[j], Yv) * self.sA[(j, k)]
        return x

    def cstream(self, st, cz, M, n):
        t = self.t; B0 = self.CB[0]; w0 = B0['w']
        if M is None:
            return (w0 * self.cls0 + B0['b2'] + st['mgc'][0].sum(0) + w0 * self.lnA(B0, B0['rb'] + st['yc'][0].sum(0)))[None]
        gc, yc = cz['mgc'][0], cz['yc'][0]
        r = (w0 * self.cls0 + B0['b2'] + gc.sum(0))[None] + t.einsum('gv,gnc->vnc', M('C1lin'), st['mgc'][0] - gc)
        return r + w0 * self.lnA(B0, (B0['rb'] + yc.sum(0))[None] + t.einsum('hv,hnc->vnc', M('C1att0'), st['yc'][0] - yc))

    # ---- forward
    def fwd(self, inp, m=None, cz=None, rec=False, probe=None, start=0, cache=None):
        """inp: dict(E (n,P,C), U (n,H,P,P), pad (n,P) bool); m: flat edge mask (NE,) or None (= all ones, clean run);
        cz: donor record (needed when m is given). Returns dict(L, st, z, h (if rec/probe))."""
        t = self.t; E, U, pad = inp['E'], inp['U'], inp['pad']; n, P, _ = E.shape
        M = None if m is None else (lambda key: m[self.ixt[key]])
        st = dict(y=[None] * 8, mg=[None] * 8, yc=[None] * 2, mgc=[None] * 2)
        if cache is not None and start > 0: st = {k: list(v) for k, v in cache.items() if isinstance(k, str)}
        hs = [None] * 10
        for k in range(start, 8):
            B = self.PB[k]
            X = self.pstream(k, st, E, cz, M, f'P{k}lin', f'P{k}att{{}}')
            if M is None:
                y = self.pattn(B, X, U, pad); xm = X + self.lnA(B, B['rb'] + y.sum(0))[None]
            else:
                Ub = cz['U'] + M(f'P{k}pair')[:, 0][None, :, None, None] * (U - cz['U'])
                y = self.pattn(B, X[:H], Ub, pad); yc = cz['y'][k]
                xm = X[H:] + self.lnA(B, (B['rb'] + yc.sum(0))[None] + t.einsum('hv,hnpc->vnpc', M(f'P{k}own'), y - yc))
            h = self.mlp(B, xm)
            if probe is not None: h = h + probe[k]
            if rec or probe is not None: hs[k] = h
            st['y'][k] = y; st['mg'][k] = self.contrib(B, h)
        padc = t.cat([t.zeros(n, 1, dtype=t.bool, device=pad.device), pad], 1)
        for b in range(2):
            if start > 8 + b: continue
            B = self.CB[b]
            Xb = self.pstream(8, st, E, cz, M, f'C{b}Xlin', f'C{b}Xatt{{}}')
            r = self.cls0.expand(n, C)[None] if b == 0 else self.cstream(st, cz, M, n)
            if M is None or b == 0:
                rh = r
            else:
                rh = r[:H]
            y = self.cattn(B, rh, Xb, padc)
            if M is None:
                rm = r + self.lnA(B, B['rb'] + y.sum(0))[None]
            else:
                yc = cz['yc'][b]; rbase = r if b == 0 else r[H:]
                rm = rbase + self.lnA(B, (B['rb'] + yc.sum(0))[None] + t.einsum('hv,hnc->vnc', M(f'C{b}own'), y - yc))
            h = self.mlp(B, rm)
            if probe is not None: h = h + probe[8 + b]
            if rec or probe is not None: hs[8 + b] = h
            st['yc'][b] = y; st['mgc'][b] = self.contrib(B, h)
        B0, B1 = self.CB; w0, w1 = B0['w'], B1['w']
        base = w0 * w1 * self.cls0 + w1 * B0['b2'] + B1['b2']
        if M is None:
            r = base + w1 * st['mgc'][0].sum(0) + st['mgc'][1].sum(0) + w0 * w1 * self.lnA(B0, B0['rb'] + st['yc'][0].sum(0)) \
                + w1 * self.lnA(B1, B1['rb'] + st['yc'][1].sum(0))
        else:
            fl = M('Flin')[:, 0][:, None, None]; f0 = M('Fatt0')[:, 0][:, None, None]; f1 = M('Fatt1')[:, 0][:, None, None]
            gcur = t.cat([st['mgc'][0], st['mgc'][1]]); gcor = t.cat([cz['mgc'][0], cz['mgc'][1]])
            sc = t.cat([w1.expand(NG, C), t.ones(NG, C, device=w1.device)])[:, None, :]
            r = base + ((gcor + fl * (gcur - gcor)) * sc).sum(0)
            r = r + w0 * w1 * self.lnA(B0, B0['rb'] + (cz['yc'][0] + f0 * (st['yc'][0] - cz['yc'][0])).sum(0))
            r = r + w1 * self.lnA(B1, B1['rb'] + (cz['yc'][1] + f1 * (st['yc'][1] - cz['yc'][1])).sum(0))
        st.pop(('lin', 8), None)
        z = self.ln(r, self.lnf)
        if M is not None: z = cz['z'] + M('z')[:, 0][None, :] * (z - cz['z'])
        L = z @ self.Wfc.T + self.bfc
        return dict(L=L, st=st, z=z, h=hs, E=E, U=U)

    def corr_of(self, out):
        return dict(E=out['E'], U=out['U'], y=out['st']['y'], mg=out['st']['mg'], yc=out['st']['yc'], mgc=out['st']['mgc'], z=out['z'])

    def prep(self, Jc, Jd):
        """clean and donor jets -> two input dicts with a common particle count P (+ the model's own logits on both)"""
        from interp.part_core import inputs, cat
        t = self.t; x, v, mk = inputs(cat([Jc, Jd]), 'full'); n = len(Jc['mask'])
        x, v, mk = (t.from_numpy(a).to(self.dev) for a in (x, v, mk)); mb = mk.bool()
        with t.no_grad():
            Emb = self.m.embed(x).masked_fill(~mb.permute(2, 0, 1), 0).permute(1, 0, 2).contiguous()
            P = v.shape[-1]; U = self.m.pair_embed(v).view(-1, H, P, P)
        pad = ~mb[:, 0]
        return (dict(E=Emb[:n], U=U[:n], pad=pad[:n]), dict(E=Emb[n:], U=U[n:], pad=pad[n:])), (x, v, mk)


def batches(J, idx, size):
    """jets sorted by multiplicity, cut into batches of at most size * 64 particle slots (size jets at P = 64)"""
    idx = np.asarray(idx); mult = J['mask'][idx].sum(1); o = np.argsort(mult, kind='stable'); out, cur = [], []
    for i in o:
        if cur and (len(cur) + 1) * max(int(mult[i]), 1) > size * 64: out.append(idx[cur]); cur = []
        cur.append(i)
    if cur: out.append(idx[cur])
    return out


def multi_eval(net, J, idx, don, masks, size=BATCH, check=False):
    """logits for every mask on jets idx with donors don[idx]: dict name -> (len(idx), 10) aligned with idx;
    also 'clean' and 'corr'. masks: dict name -> torch flat mask"""
    from interp.part_core import sub
    t = net.t; pos = {int(i): r for r, i in enumerate(idx)}; out = {k: np.zeros((len(idx), 10), np.float32) for k in list(masks) + ['clean', 'corr']}
    md = dict(clean_vs_model=0.0, ones_vs_clean=0.0, zeros_vs_donor=0.0)
    with t.no_grad():
        for bi in batches(J, np.asarray(idx), size):
            (ic, idn), raw = net.prep(sub(J, bi), sub(J, don[bi])); rows = [pos[int(i)] for i in bi]
            oc = net.fwd(ic); od = net.fwd(idn); cz = net.corr_of(od)
            out['clean'][rows] = oc['L'].cpu().numpy(); out['corr'][rows] = od['L'].cpu().numpy()
            if check:
                Lm = net.m(*(a[:len(bi)] for a in raw)).float()
                md['clean_vs_model'] = max(md['clean_vs_model'], float((Lm - oc['L']).abs().max()))
                one = t.ones(net.G.NE, device=net.dev); zero = t.zeros(net.G.NE, device=net.dev)
                md['ones_vs_clean'] = max(md['ones_vs_clean'], float((net.fwd(ic, one, cz)['L'] - oc['L']).abs().max()))
                md['zeros_vs_donor'] = max(md['zeros_vs_donor'], float((net.fwd(ic, zero, cz)['L'] - od['L']).abs().max()))
            for k, mk in masks.items():
                out[k][rows] = net.fwd(ic, mk, cz)['L'].cpu().numpy()
    if check: out['_check'] = md
    return out


# ------------------------------------------------------------------------------------------------------------ metrics
class Scorer:
    def __init__(self, y, iW, iZ, Lclean, Lcorr, rng):
        self.y, self.iW, self.iZ = y, iW, iZ; self.s = np.where(y == iW, 1.0, -1.0)
        self.wz = np.isin(y, [iW, iZ]); self.dcl = Lclean[:, iW] - Lclean[:, iZ]; self.dco = Lcorr[:, iW] - Lcorr[:, iZ]
        self.Lclean = Lclean; self.Wb = boot_W(len(y), NB, rng); self.den = self.s * (self.dcl - self.dco)
        self.call0 = np.where(self.dcl > 0, iW, iZ); self.ok0 = self.call0 == y

    def F(self, L):
        d = L[:, self.iW] - L[:, self.iZ]; num = self.s * (d - self.dco)
        f = num.sum() / self.den.sum(); fb = (self.Wb @ num) / (self.Wb @ self.den); return float(f), fb

    def summary(self, L, full=True):
        y, wz = self.y, self.wz; d = L[:, self.iW] - L[:, self.iZ]; call = np.where(d > 0, self.iW, self.iZ); ok = call == y
        f, fb = self.F(L); accb = (self.Wb[:, wz] @ ok[wz]) / self.Wb[:, wz].sum(1)
        e = dict(F=f, F_ci=ci(fb), wz_acc=float(ok[wz].mean()), wz_acc_ci=ci(accb), wz_agree_full=float((call == self.call0)[wz].mean()),
                 acc10=float((L.argmax(1) == y).mean()), mcnemar_vs_full=mcnemar(self.ok0[wz], ok[wz]))
        e['_fb'] = fb; e['_accb'] = accb
        return e


def jetlevel_faith(L, Lcl, Lco, iW, iZ):
    d = L[:, iW] - L[:, iZ]; dcl = Lcl[:, iW] - Lcl[:, iZ]; dco = Lco[:, iW] - Lco[:, iZ]
    return float(1 - np.abs(d - dcl).mean() / np.abs(dco - dcl).mean())


def confusion(L, y):
    cm = np.zeros((10, 10), int); np.add.at(cm, (y, L.argmax(1)), 1); return cm


# ------------------------------------------------------------------------------------------------------------ main
def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from interp.part_core import kin, build, jmass, sub, cat, sample, CL, PRE, N98
    OUT.mkdir(parents=True, exist_ok=True)
    iW, iZ = CL.index('Wqq'), CL.index('Zqq')
    G = Graph(); log(f'graph: {G.NE} edges, {len(G.nodes)} writer nodes')
    res = dict(smoke=SMOKE, NPER=NPER, NALL=NALL, NUNIT=NUNIT, NB=NB, NRAND=NRAND, NACDC=NACDC, ACDC_TOP=ACDC_TOP, NFID=NFID, TAUS=TAUS,
               ALPHAS=ALPHAS, BATCH=BATCH, n_edges=G.NE, n_nodes=len(G.nodes), val_dir=str(VAL),
               label_map=dict(class_heads='G-series bXhY = C{X-1}H{Y-1}; b2h8 = C1H7, b1h3 = C0H2', neuron='nK = z{K-1}; n98 = z97',
                              particle='P{k}H{h} = particle block k+1, head h+1 (0-based in names); P{k}B{h} = pair bias of that head; '
                                       'P{k}M{g} / C{b}M{g} = MLP group g (32 units ranked by |attribution|, g = 0 is the top 32)'))
    # ---------------- samples
    files = {}
    rngD, rngE, rngA = np.random.default_rng(2026), np.random.default_rng(2027), np.random.default_rng(2028)
    parts_D, parts_E = [], []
    for c in ('Wqq', 'Zqq'):
        import uproot
        p = VAL / f'{PRE[c]}_123.root'; N = uproot.open(p)['tree'].num_entries
        rD = np.sort(rngD.choice(N, NPER, replace=False)); rest = np.setdiff1d(np.arange(N), rD); rE = np.sort(rngE.choice(rest, NPER, replace=False))
        (JD, JE), _ = load_sel(p, [rD, rE]); parts_D.append(JD); parts_E.append(JE)
        files[c] = dict(file=p.name, entries=N, disc_first_rows=rD[:5].tolist(), eval_first_rows=rE[:5].tolist(), overlap=int(np.intersect1d(rD, rE).size))
    pA = []
    for c in CL:
        import uproot
        p = VAL / f'{PRE[c]}_124.root'; N = uproot.open(p)['tree'].num_entries; r = np.sort(rngA.choice(N, NALL, replace=False))
        (JA,), _ = load_sel(p, [r]); pA.append(JA); files[f'all_{c}'] = dict(file=p.name, entries=N, first_rows=r[:5].tolist())
    res['samples'] = files

    def prepset(parts):
        J0 = cat(parts); J = build(J0, *kin(J0)); return J, J0['label'].astype(int), jmass(J)
    JD, yD, mD = prepset(parts_D); JE, yE, mE = prepset(parts_E); JA, yA, mA = prepset(pA); del parts_D, parts_E, pA
    for nm, y_ in (('disc', yD), ('eval', yE), ('all', yA)): log(f'{nm}: {len(y_)} jets, classes {np.bincount(y_, minlength=10).tolist()}')
    rng = np.random.default_rng(2026)
    donD = donors_B(JD['jet_pt'], yD, iW, iZ, rng); donE = donors_B(JE['jet_pt'], yE, iW, iZ, rng)
    donEA, nfb = donors_A(mE, JE['mask'].sum(1), yE, rng); donAU = donors_U(len(yA), rng)
    res['donors'] = dict(B_eval_rel_dpt_median=float(np.median(np.abs(JE['jet_pt'][donE] - JE['jet_pt']) / JE['jet_pt'])),
                         B_eval_dm_median=float(np.median(mE[donE] - mE)), A_eval_absdm_median=float(np.median(np.abs(mE[donEA] - mE))),
                         A_eval_dmult_median=float(np.median(np.abs(JE['mask'].sum(1)[donEA] - JE['mask'].sum(1)))), A_fallback=nfb,
                         U_all_same_class=float((yA[donAU] == yA).mean()))
    log(f'donors: {res["donors"]}')

    net = Net(G)
    # P1 sample check
    JP = sample(100 if SMOKE else 1000, 3000); RP = net.net.run(build(JP, *kin(JP)))
    res['P1_sample_acc'] = float((RP['logits'].argmax(1) == JP['label']).mean()); del JP, RP
    log(f'ParT_full P1-sample accuracy {res["P1_sample_acc"]:.4f} (expected ~0.859)')

    # ---------------- step 0: unit attribution -> MLP groups
    t1 = time.time(); sc_u = [np.zeros(4 * C) for _ in range(10)]; sD = np.where(yD == iW, 1.0, -1.0)
    uidx = np.sort(np.random.default_rng(7).choice(len(yD), min(NUNIT, len(yD)), replace=False))
    for bi in batches(JD, uidx, IGB):
        (ic, idn), _ = net.prep(sub(JD, bi), sub(JD, donD[bi]))
        with torch.no_grad(): hd = net.fwd(idn, rec=True)['h']
        probe = [torch.zeros(len(bi), ic['E'].shape[1], 4 * C, device=net.dev, requires_grad=True) for _ in range(8)] + \
                [torch.zeros(len(bi), 4 * C, device=net.dev, requires_grad=True) for _ in range(2)]
        o = net.fwd(ic, probe=probe); s = torch.from_numpy(sD[bi]).float().to(net.dev)
        Dm = (s * (o['L'][:, iW] - o['L'][:, iZ])).sum(); gr = torch.autograd.grad(Dm, probe)
        for k in range(10):
            dims = tuple(range(gr[k].dim() - 1))
            sc_u[k] += ((hd[k] - o['h'][k].detach()) * gr[k]).sum(dims).cpu().numpy()
    perms = [np.argsort(-np.abs(s_), kind='stable') for s_ in sc_u]; net.set_perm(perms)
    res['unit_attribution'] = {('P%d' % k if k < 8 else 'C%d' % (k - 8)): dict(top32_share=float(np.abs(s_)[p[:32]].sum() / np.abs(s_).sum()),
                                                                            total=float(s_.sum()), top5=[int(i) for i in p[:5]])
                               for k, (s_, p) in enumerate(zip(sc_u, perms))}
    log(f'step 0 unit attribution ({time.time() - t1:.0f} s): top-32 share per block ' +
        ', '.join(f'{k} {v["top32_share"]:.2f}' for k, v in res['unit_attribution'].items()))

    # ---------------- checks on eval + baseline
    t1 = time.time()
    chk = multi_eval(net, JE, np.arange(0, len(yE), max(1, len(yE) // 400)), donE, {}, check=True)
    res['exactness'] = chk['_check']; log(f'exactness (|dlogit| max): {res["exactness"]} ({time.time() - t1:.0f} s)')

    # ---------------- step 1: EAP-IG on discovery
    t1 = time.time(); ig = np.zeros(G.NE); eap = np.zeros(G.NE); Dcl = Dco = 0.0; peak = 0
    for nb_, bi in enumerate(batches(JD, np.arange(len(yD)), IGB)):
        (ic, idn), _ = net.prep(sub(JD, bi), sub(JD, donD[bi])); s = torch.from_numpy(sD[bi]).float().to(net.dev)
        with torch.no_grad():
            od = net.fwd(idn); cz = net.corr_of(od); oc = net.fwd(ic)
        Dcl += float((s * (oc['L'][:, iW] - oc['L'][:, iZ])).sum()); Dco += float((s * (od['L'][:, iW] - od['L'][:, iZ])).sum())
        for a in ALPHAS + [1.0]:
            mm = torch.full((G.NE,), a, device=net.dev, requires_grad=True)
            L = net.fwd(ic, mm, cz)['L']; g, = torch.autograd.grad((s * (L[:, iW] - L[:, iZ])).sum(), mm)
            if a < 1: ig += g.double().cpu().numpy() / len(ALPHAS)
            else: eap += g.double().cpu().numpy()
        if torch.cuda.is_available(): peak = max(peak, torch.cuda.max_memory_allocated() / 2 ** 30)
        if nb_ % 20 == 0: log(f'  EAP-IG batch {nb_}: {(time.time() - t1):.0f} s, peak GPU {peak:.1f} GiB')
    nD = len(yD); ig /= nD; eap /= nD; dtot = (Dcl - Dco) / nD
    res['eap'] = dict(D_clean_minus_corr_per_jet=dtot, sum_ig=float(ig.sum()), sum_eap=float(eap.sum()), completeness_ratio=float(ig.sum() / dtot),
                      seconds=time.time() - t1, peak_gpu_gib=peak)
    rank = np.argsort(-np.abs(ig), kind='stable'); absn = np.abs(ig) / np.abs(ig).sum()
    res['eap']['abs_share_top'] = {str(k): float(absn[rank[:k]].sum()) for k in (10, 100, 1000, 2000, 5000)}
    res['eap']['top30'] = [dict(edge=f'{G.W[e]}->{G.R[e]}', ig=float(ig[e]), eap=float(eap[e]), share=float(ig[e] / dtot)) for e in rank[:30]]
    from scipy.stats import spearmanr
    res['eap']['spearman_ig_eap_abs'] = float(spearmanr(np.abs(ig), np.abs(eap)).correlation)
    log(f'step 1 EAP-IG ({time.time() - t1:.0f} s): D(1)-D(0) per jet {dtot:.3f}, sum IG {ig.sum():.3f} (ratio {ig.sum() / dtot:.3f}), '
        f'sum EAP {eap.sum():.3f}; |IG| share top-100 {res["eap"]["abs_share_top"]["100"]:.3f}, top-1000 {res["eap"]["abs_share_top"]["1000"]:.3f}')
    log('  top edges: ' + '; '.join(f'{d["edge"]} {d["share"]:+.3f}' for d in res['eap']['top30'][:10]))
    nscore = {nd: 0.0 for nd in G.nodes}
    for e in range(G.NE): nscore[G.W[e]] += ig[e]
    nodes_ranked = sorted(G.nodes, key=lambda nd: -abs(nscore[nd]))
    res['node_scores_top30'] = [dict(node=nd, score=nscore[nd], share=nscore[nd] / dtot) for nd in nodes_ranked[:30]]
    log('  top nodes: ' + ', '.join(f'{d["node"]} {d["share"]:+.3f}' for d in res['node_scores_top30'][:12]))
    np.savez(OUT / ('p9_eap_scores_smoke.npz' if SMOKE else 'p9_eap_scores.npz'), ig=ig, eap=eap, W=G.W, R=G.R, perms=np.array(perms))

    # ---------------- step 2: ACDC on cached discovery subset
    t1 = time.time(); top = rank[:ACDC_TOP]; m = np.zeros(G.NE, np.float32); m[top] = 1
    aidx = np.sort(np.random.default_rng(11).choice(len(yD), min(NACDC, len(yD)), replace=False)); cache = []
    with torch.no_grad():
        for bi in batches(JD, aidx, 64):
            (ic, idn), _ = net.prep(sub(JD, bi), sub(JD, donD[bi])); od = net.fwd(idn); cz = net.corr_of(od); oc = net.fwd(ic)
            s = torch.from_numpy(sD[bi]).float().to(net.dev)
            cache.append(dict(ic=ic, cz=cz, s=s, dcl=oc['L'][:, iW] - oc['L'][:, iZ], dco=od['L'][:, iW] - od['L'][:, iZ], st=None))
        den = sum(float((c['s'] * (c['dcl'] - c['dco'])).sum()) for c in cache)

        def acdc_F(mt, start, commit=False):
            num, sts = 0.0, []
            for c in cache:
                o = net.fwd(c['ic'], mt, c['cz'], start=start, cache=c['st']); sts.append(o['st'])
                num += float((c['s'] * (o['L'][:, iW] - o['L'][:, iZ] - c['dco'])).sum())
            return num / den, sts
        mt = torch.from_numpy(m).to(net.dev); Fc, sts = acdc_F(mt, 0)
        for c, s_ in zip(cache, sts): c['st'] = s_
        res['acdc'] = dict(start_F=Fc, start_edges=int(m.sum()), passes=[])
        log(f'step 2 ACDC start: top {ACDC_TOP} edges, F = {Fc:.4f} on {len(aidx)} discovery jets')
        order = sorted(top.tolist(), key=lambda e: (-G.rstage[e], abs(ig[e])))
        circuits = {}
        for tau in TAUS:
            tp = time.time(); nrem = 0
            for e in order:
                if m[e] == 0: continue
                mt[int(e)] = 0.0; Fn, sts = acdc_F(mt, int(G.rstage[e]))
                if Fc - Fn < tau:
                    m[e] = 0; Fc = Fn; nrem += 1
                    for c, s_ in zip(cache, sts): c['st'] = s_
                else:
                    mt[int(e)] = 1.0
            nodes = sorted({G.W[e] for e in np.where(m > 0)[0]} | {G.R[e] for e in np.where(m > 0)[0]} - {'OUT', 'LOGIT'}, key=G.node_order)
            circuits[tau] = dict(edges=np.where(m > 0)[0].copy(), nodes=nodes)
            res['acdc']['passes'].append(dict(tau=tau, removed=nrem, edges=int(m.sum()), nodes=len(nodes), F_disc=Fc, seconds=time.time() - tp))
            log(f'  ACDC tau {tau}: removed {nrem}, kept {int(m.sum())} edges / {len(nodes)} nodes, F(disc subset) {Fc:.4f} ({time.time() - tp:.0f} s)')
    del cache; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---------------- masks for evaluation
    tm = lambda a: torch.from_numpy(np.asarray(a, np.float32)).to(net.dev)
    K5 = int(round(0.05 * G.NE))
    EK = sorted(set([10, 30, 100, 300, 1000, K5, 2000, 4000, 8000, 15000, G.NE]))
    NK = [2, 4, 8, 16, 32, 64, 128, 256, len(G.nodes)]
    masks = {}
    for k in EK:
        a = np.zeros(G.NE, np.float32); a[rank[:k]] = 1; masks[f'edge_top{k}'] = tm(a)
    for k in NK: masks[f'node_top{k}'] = tm(G.node_mask(nodes_ranked[:k]))
    for tau, cc in circuits.items():
        a = np.zeros(G.NE, np.float32); a[cc['edges']] = 1; masks[f'acdc{tau}_edges'] = tm(a); masks[f'acdc{tau}_literal'] = tm(G.node_mask(cc['nodes']))
        masks[f'acdc{tau}_noderank'] = tm(G.node_mask(nodes_ranked[:len(cc['nodes'])]))
    pnodes = [nd for nd in G.nodes if nd[0] == 'P' or nd == 'E']
    REF = {'lit_b2h8_b1h3_n98': ['C1H7', 'C0H2', 'z97'], 'lit_16heads_n98': [f'C{b}H{h}' for b in range(2) for h in range(H)] + ['z97']}
    REF.update({'p3scope_' + k[4:]: pnodes + v for k, v in list(REF.items())})
    REF['p3scope_classstage_ablated'] = pnodes
    for k, v in REF.items(): masks['ref_' + k] = tm(G.node_mask(v))
    masks['all_ablated'] = tm(np.zeros(G.NE)); masks['full'] = tm(np.ones(G.NE))
    res['circuits'] = {k: dict(n_edges=int(v.sum().item())) for k, v in masks.items()}
    for tau, cc in circuits.items():
        res['circuits'][f'acdc{tau}_edges']['n_nodes'] = len(cc['nodes']); res['circuits'][f'acdc{tau}_literal']['n_nodes'] = len(cc['nodes'])
        res['circuits'][f'acdc{tau}_noderank']['n_nodes'] = len(cc['nodes'])
        res['circuits'][f'acdc{tau}_edges']['nodes'] = cc['nodes']
        res['circuits'][f'acdc{tau}_edges']['node_categories'] = {cat_: int(sum(G.ncat(nd) == cat_ for nd in cc['nodes'])) for cat_ in sorted({G.ncat(nd) for nd in cc['nodes']})}
    for k in NK: res['circuits'][f'node_top{k}']['nodes'] = nodes_ranked[:k] if k <= 64 else None

    # ---------------- evaluation set, corruption B
    t1 = time.time(); allE = np.arange(len(yE))
    LE = multi_eval(net, JE, allE, donE, masks)
    SB = Scorer(yE, iW, iZ, LE['clean'], LE['corr'], np.random.default_rng(5))
    res['eval_B'] = dict(unedited_wz_acc=float(SB.ok0.mean()), unedited_acc10=float((LE['clean'].argmax(1) == yE).mean()),
                         corr_wz_acc=float((np.where(SB.dco > 0, iW, iZ) == yE).mean()))
    R = {}
    for k in masks:
        R[k] = SB.summary(LE[k]); R[k]['n_edges'] = res['circuits'][k]['n_edges']
    res['eval_B']['circuits'] = R
    log(f'eval B ({time.time() - t1:.0f} s): unedited W/Z acc {res["eval_B"]["unedited_wz_acc"]:.4f}, all-ablated {R["all_ablated"]["wz_acc"]:.4f}, '
        f'full mask F {R["full"]["F"]:.4f}')
    for k in masks: log(f'  {k:28s} edges {R[k]["n_edges"]:6d}  F {R[k]["F"]:+.3f} {np.round(R[k]["F_ci"], 3).tolist()}  W/Z acc {R[k]["wz_acc"]:.4f}')
    # hypothesis at 5 % of edges
    e5 = R[f'edge_top{K5}']; best_small = max([k for k in masks if k.startswith('acdc') and k.endswith('_edges') and R[k]['n_edges'] <= K5] or [f'edge_top{K5}'],
                                               key=lambda k: R[k]['F'])
    b5 = R[best_small]
    res['hypothesis'] = dict(
        H0='the W-vs-Z logit difference is not recovered to F >= 0.8 by any circuit with <= 5 % of edges (distributed)',
        H1='a circuit with <= 5 % of edges recovers F >= 0.8 and W/Z accuracy within 3 points of the model (localized)',
        K5=K5, F_top5pct=e5['F'], F_top5pct_ci=e5['F_ci'], p_F_top5pct_le_0p8=float(max(1 / NB, (e5['_fb'] <= 0.8).mean())),
        best_small_circuit=best_small, F_best_small=b5['F'], F_best_small_ci=b5['F_ci'], p_F_best_small_le_0p8=float(max(1 / NB, (b5['_fb'] <= 0.8).mean())),
        wz_acc_top5pct=e5['wz_acc'], wz_acc_drop_top5pct=float(SB.ok0.mean() - e5['wz_acc']),
        edges_needed_F0p8=next((k for k in EK if R[f'edge_top{k}']['F'] >= 0.8), None),
        edges_needed_F0p5=next((k for k in EK if R[f'edge_top{k}']['F'] >= 0.5), None),
        nodes_needed_F0p8=next((k for k in NK if R[f'node_top{k}']['F'] >= 0.8), None),
        nodes_needed_wzacc_within3=next((k for k in NK if R[f'node_top{k}']['wz_acc'] >= SB.ok0.mean() - 0.03), None))
    log(f'hypothesis: {clean({k: v for k, v in res["hypothesis"].items() if not k.startswith("H")})}')

    # ---------------- random circuits (eval[::5])
    t1 = time.time(); sub5 = np.arange(0, len(yE), 10); rr = np.random.default_rng(13)          # eval[::10]: rows are W then Z, so both classes are sampled
    cats = {}
    for e, c_ in enumerate(G.ecat): cats.setdefault(c_, []).append(e)
    cats = {k: np.array(v) for k, v in cats.items()}
    ncats = {}
    for nd in G.nodes: ncats.setdefault(G.ncat(nd), []).append(nd)
    rmasks, rinfo = {}, {}
    for tau, cc in circuits.items():
        ce = cc['edges']
        for r_ in range(NRAND):
            a = np.zeros(G.NE, np.float32)
            for c_ in np.unique(G.ecat[ce]):
                a[rr.choice(cats[c_], int((G.ecat[ce] == c_).sum()), replace=False)] = 1
            rmasks[f'rand_e{tau}_{r_}'] = tm(a)
            nodes = []
            for c_ in sorted({G.ncat(nd) for nd in cc['nodes']}):
                kk = sum(G.ncat(nd) == c_ for nd in cc['nodes']); nodes += list(rr.choice(ncats[c_], kk, replace=False))
            rmasks[f'rand_n{tau}_{r_}'] = tm(G.node_mask(nodes))
        rmasks[f'acdc{tau}_edges'] = masks[f'acdc{tau}_edges']; rmasks[f'acdc{tau}_literal'] = masks[f'acdc{tau}_literal']
    for k in REF: rmasks['ref_' + k] = masks['ref_' + k]
    for k in REF:
        if k.startswith('lit_'):
            for r_ in range(NRAND):
                nodes = []
                for nd in REF[k]:
                    nodes.append(rr.choice([x for x in ncats[G.ncat(nd)] if x not in nodes]))
                rmasks[f'rand_{k}_{r_}'] = tm(G.node_mask(nodes))
    LR = multi_eval(net, JE, sub5, donE, rmasks)
    S5 = Scorer(yE[sub5], iW, iZ, LR['clean'], LR['corr'], np.random.default_rng(6)); RC = {}
    for tau in circuits:
        for kind, pre in (('edges', 'rand_e'), ('literal', 'rand_n')):
            k = f'acdc{tau}_{kind}'; e = S5.summary(LR[k])
            Fr = np.array([S5.F(LR[f'{pre}{tau}_{r_}'])[0] for r_ in range(NRAND)])
            Ar = np.array([(np.where(LR[f'{pre}{tau}_{r_}'][:, iW] > LR[f'{pre}{tau}_{r_}'][:, iZ], iW, iZ) == yE[sub5]).mean() for r_ in range(NRAND)])
            RC[k] = dict(F=e['F'], F_ci=e['F_ci'], wz_acc=e['wz_acc'], random_F_mean=float(Fr.mean()), random_F_ci=ci(Fr), random_F_max=float(Fr.max()),
                         random_wz_acc_mean=float(Ar.mean()), random_wz_acc_max=float(Ar.max()), random_F_values=Fr, random_wz_acc_values=Ar,
                         p_perm_F=float((1 + (Fr >= e['F']).sum()) / (1 + NRAND)), p_perm_acc=float((1 + (Ar >= e['wz_acc']).sum()) / (1 + NRAND)))
            log(f'  random vs {k}: F {e["F"]:.3f} vs random mean {Fr.mean():.3f} max {Fr.max():.3f}; acc {e["wz_acc"]:.4f} vs {Ar.mean():.4f} (p {RC[k]["p_perm_F"]:.3f})')
    for k in REF:
        if k.startswith('lit_'):
            e = S5.summary(LR['ref_' + k]); Ar = np.array([(np.where(LR[f'rand_{k}_{r_}'][:, iW] > LR[f'rand_{k}_{r_}'][:, iZ], iW, iZ) == yE[sub5]).mean() for r_ in range(NRAND)])
            Fr = np.array([S5.F(LR[f'rand_{k}_{r_}'])[0] for r_ in range(NRAND)])
            RC['ref_' + k] = dict(F=e['F'], wz_acc=e['wz_acc'], random_F_mean=float(Fr.mean()), random_wz_acc_mean=float(Ar.mean()), random_wz_acc_max=float(Ar.max()),
                                  p_perm_acc=float((1 + (Ar >= e['wz_acc']).sum()) / (1 + NRAND)), p_perm_F=float((1 + (Fr >= e['F']).sum()) / (1 + NRAND)))
    res['random_controls'] = dict(n_jets=len(sub5), NRAND=NRAND, circuits=RC, seconds=time.time() - t1)
    log(f'random controls done ({time.time() - t1:.0f} s)')
    del rmasks, LR

    # ---------------- attribution fidelity
    t1 = time.time(); others = rank[NFID:]; fid_e = list(rank[:NFID]) + list(np.random.default_rng(17).choice(others, NFID, replace=False))
    fm = {}
    for e in fid_e:
        a = torch.ones(G.NE, device=net.dev); a[int(e)] = 0; fm[f'e{e}'] = a
    LF = multi_eval(net, JE, sub5, donE, fm); s5 = np.where(yE[sub5] == iW, 1.0, -1.0); d0 = LF['clean'][:, iW] - LF['clean'][:, iZ]
    Wb5 = boot_W(len(sub5), NB, np.random.default_rng(19)); act, pred_ig, pred_eap, pv, act_ci = [], [], [], [], []
    for e in fid_e:
        L = LF[f'e{e}']; dd = s5 * ((L[:, iW] - L[:, iZ]) - d0); a = float(dd.mean()); bs = (Wb5 @ dd) / len(sub5)
        act.append(a); act_ci.append(ci(bs)); pv.append(float(max(1 / NB, min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))))
        pred_ig.append(-float(ig[e])); pred_eap.append(-float(eap[e]))
    act, pred_ig, pred_eap = map(np.array, (act, pred_ig, pred_eap)); ph = holm(pv)
    top_ = slice(0, NFID)
    res['fidelity'] = dict(
        n_jets=len(sub5), edges=[f'{G.W[e]}->{G.R[e]}' for e in fid_e], rank=[int(np.where(rank == e)[0][0]) for e in fid_e], actual=act, actual_ci=act_ci,
        pred_ig=pred_ig, pred_eap=pred_eap, p=pv, p_holm=ph,
        spearman_ig_all=float(spearmanr(pred_ig, act).correlation), spearman_eap_all=float(spearmanr(pred_eap, act).correlation),
        spearman_ig_top=float(spearmanr(pred_ig[top_], act[top_]).correlation), spearman_eap_top=float(spearmanr(pred_eap[top_], act[top_]).correlation),
        sign_agree_ig_top=float((np.sign(pred_ig[top_]) == np.sign(act[top_])).mean()),
        ratio_actual_over_ig_top_median=float(np.median(act[top_] / np.where(pred_ig[top_] == 0, np.nan, pred_ig[top_]))),
        n_top_holm_sig=int(sum(p < 0.05 for p in ph[:NFID])), n_rand_holm_sig=int(sum(p < 0.05 for p in ph[NFID:])),
        seconds=time.time() - t1)
    log(f'fidelity ({time.time() - t1:.0f} s): Spearman IG vs actual (top {NFID}) {res["fidelity"]["spearman_ig_top"]:.3f}, (all {2 * NFID}) '
        f'{res["fidelity"]["spearman_ig_all"]:.3f}; EAP top {res["fidelity"]["spearman_eap_top"]:.3f}; sign agree {res["fidelity"]["sign_agree_ig_top"]:.2f}; '
        f'median actual/IG {res["fidelity"]["ratio_actual_over_ig_top_median"]:.2f}; Holm-significant top {res["fidelity"]["n_top_holm_sig"]}, random {res["fidelity"]["n_rand_holm_sig"]}')
    del LF, fm

    # ---------------- corruption A (eval[::5])
    t1 = time.time(); amk = {k: masks[k] for k in masks if k.startswith(('acdc', 'ref_', 'all_ablated')) or k in (f'edge_top{K5}', 'edge_top2000')}
    subA = np.arange(0, len(yE), 5)
    LA = multi_eval(net, JE, subA, donEA, amk); ya = yE[subA]; okc = np.where(LA['clean'][:, iW] > LA['clean'][:, iZ], iW, iZ) == ya
    RA = {}
    for k in amk:
        call = np.where(LA[k][:, iW] > LA[k][:, iZ], iW, iZ)
        RA[k] = dict(wz_acc=float((call == ya).mean()), jet_faith=jetlevel_faith(LA[k], LA['clean'], LA['corr'], iW, iZ), mcnemar_vs_full=mcnemar(okc, call == ya))
    res['eval_A'] = dict(n_jets=len(subA), unedited_wz_acc=float(okc.mean()), circuits=RA, seconds=time.time() - t1)
    log(f'corruption A ({time.time() - t1:.0f} s): ' + '; '.join(f'{k} {v["wz_acc"]:.3f}/{v["jet_faith"]:.2f}' for k, v in RA.items()))

    # ---------------- 10-class read (all-class set, uniform donors)
    t1 = time.time(); cmk = {k: masks[k] for k in masks if (k.startswith(('acdc', 'ref_', 'all_ablated', 'full')) and 'noderank' not in k) or k in (f'edge_top{K5}', 'node_top64', 'node_top128')}
    LC = multi_eval(net, JA, np.arange(len(yA)), donAU, cmk); ok10 = LC['clean'].argmax(1) == yA; wzA = np.isin(yA, [iW, iZ])
    WbA = boot_W(len(yA), NB, np.random.default_rng(23))
    RT = dict(unedited=dict(acc10=float(ok10.mean()), acc10_ci=ci((WbA @ ok10) / len(yA)),
                            wz_acc=float((np.where(LC['clean'][:, iW] > LC['clean'][:, iZ], iW, iZ) == yA)[wzA].mean()), confusion=confusion(LC['clean'], yA)))
    for k in cmk:
        L = LC[k]; ok = L.argmax(1) == yA; cw = np.where(L[:, iW] > L[:, iZ], iW, iZ)
        RT[k] = dict(acc10=float(ok.mean()), acc10_ci=ci((WbA @ ok) / len(yA)), wz_acc=float((cw == yA)[wzA].mean()),
                     agree_full=float((L.argmax(1) == LC['clean'].argmax(1)).mean()), follows_donor_class=float((L.argmax(1) == yA[donAU]).mean()),
                     mcnemar_vs_full=mcnemar(ok10, ok), confusion=confusion(L, yA))
    res['ten_class'] = dict(n_jets=len(yA), donors='uniform over the all-class set', circuits=RT, seconds=time.time() - t1)
    log(f'10-class ({time.time() - t1:.0f} s): unedited {RT["unedited"]["acc10"]:.4f}; ' + '; '.join(f'{k} {v["acc10"]:.3f} (W/Z {v["wz_acc"]:.3f})' for k, v in RT.items() if k != 'unedited'))

    json.dump(clean(res), open(FN, 'w'), indent=1); log(f'wrote {FN}')
    try: plot(clean(res))
    except Exception as ex: log(f'plot failed: {ex!r}')
    log('P9 DONE')


# ------------------------------------------------------------------------------------------------------------ figure
def plot(res=None):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    if res is None: res = json.load(open(FN))
    R = res['eval_B']['circuits']; NE = res['n_edges']; acc0 = res['eval_B']['unedited_wz_acc']
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), dpi=150)
    a = ax[0]
    ek = sorted([k for k in R if k.startswith('edge_top')], key=lambda k: R[k]['n_edges'])
    x = np.array([R[k]['n_edges'] for k in ek]); f = np.array([R[k]['F'] for k in ek]); lo = np.array([R[k]['F_ci'][0] for k in ek]); hi = np.array([R[k]['F_ci'][1] for k in ek])
    a.fill_between(x, lo, hi, color='#2c7fb8', alpha=0.25, lw=0); a.plot(x, f, 'o-', color='#2c7fb8', ms=3, label='top-k edges by attribution (EAP-IG)')
    RC = res['random_controls']['circuits']
    for i, tau in enumerate(res['TAUS']):
        k = f'acdc{tau}_edges'
        if k in R:
            a.scatter([R[k]['n_edges']], [R[k]['F']], color='#c0392b', zorder=4, s=30, label='ACDC-pruned circuits' if i == 0 else None)
            if k in RC:
                rv = np.array(RC[k]['random_F_values'], float)
                a.scatter(np.full(len(rv), R[k]['n_edges']) * np.exp(np.random.default_rng(i).uniform(-0.08, 0.08, len(rv))), rv, s=4, color='#999999',
                          label='random circuits, same size and layout' if i == 0 else None)
    a.axhline(0.8, color='k', ls='--', lw=0.8); a.axvline(0.05 * NE, color='k', ls=':', lw=0.8); a.text(0.05 * NE * 1.08, 0.02, '5 % of edges', fontsize=7)
    a.set_xscale('log'); a.set_xlabel('edges kept (of %d)' % NE); a.set_ylabel('faithfulness F (1 = full model, 0 = all resampled)')
    a.set_ylim(-0.2, 1.1); a.set_title('(a) W-vs-Z logit difference recovered\nvs edges kept (rest from W<->Z donor)', fontsize=10); a.legend(fontsize=7, loc='upper left')
    a = ax[1]
    nk = sorted([k for k in R if k.startswith('node_top')], key=lambda k: int(k[8:]))
    xn = np.array([int(k[8:]) for k in nk]); acc = np.array([R[k]['wz_acc'] for k in nk]) * 100
    alo = np.array([R[k]['wz_acc_ci'][0] for k in nk]) * 100; ahi = np.array([R[k]['wz_acc_ci'][1] for k in nk]) * 100
    a.fill_between(xn, alo, ahi, color='#2c7fb8', alpha=0.25, lw=0); a.plot(xn, acc, 'o-', color='#2c7fb8', ms=3, label='top-k nodes, literal circuit')
    for i, tau in enumerate(res['TAUS']):
        k = f'acdc{tau}_literal'
        if k in R: a.scatter([res['circuits'][k]['n_nodes']], [R[k]['wz_acc'] * 100], color='#c0392b', s=30, zorder=4, label='ACDC circuits, literal' if i == 0 else None)
    mk = {'ref_lit_b2h8_b1h3_n98': ('s', '{b2h8, b1h3, n98}'), 'ref_lit_16heads_n98': ('^', '16 class heads + n98')}
    for k, (mm, lb) in mk.items():
        if k in R: a.scatter([3 if 'b2h8' in k else 17], [R[k]['wz_acc'] * 100], marker=mm, color='k', s=30, label=lb + ' (P3)')
    a.axhline(acc0 * 100, color='k', ls='--', lw=0.8, label=f'full model {acc0 * 100:.1f} %')
    a.axhline(R['all_ablated']['wz_acc'] * 100, color='k', ls=':', lw=0.8, label=f'everything resampled {R["all_ablated"]["wz_acc"] * 100:.1f} %')
    a.set_xscale('log'); a.set_xlabel('nodes kept (heads, MLP groups, pair biases, z coordinates; of %d)' % res['n_nodes']); a.set_ylabel('W vs Z accuracy on true W/Z jets [%]')
    a.set_title('(b) literal circuit: only the kept nodes, everything else\nresampled from a W<->Z donor', fontsize=10); a.legend(fontsize=7, loc='upper left')
    a = ax[2]; fd = res['fidelity']; n = len(fd['actual']) // 2
    pa, ac = np.array(fd['pred_ig'], float), np.array(fd['actual'], float)
    a.scatter(pa[:n], ac[:n], s=12, color='#c0392b', label=f'top {n} edges (Spearman {fd["spearman_ig_top"]:.2f})')
    a.scatter(pa[n:], ac[n:], s=12, color='#999999', label=f'{n} random edges')
    lim = max(np.abs(pa).max(), np.abs(ac).max()) * 1.1; a.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8)
    a.set_xlabel('predicted change of the W-vs-Z logit difference per jet\n(EAP-IG, discovery jets)'); a.set_ylabel('measured change when the edge alone is resampled\n(evaluation jets)')
    a.set_title('(c) attribution fidelity, single edges', fontsize=10); a.legend(fontsize=7)
    fig.suptitle('ParT_full W vs Z: faithfulness of whole-network circuits vs size (P9)', fontsize=11)
    fig.tight_layout(); p = OUT / ('p9_eap_wz_smoke.png' if res.get('smoke') else 'p9_eap_wz.png'); fig.savefig(p, bbox_inches='tight'); log(f'wrote {p}')


def post():
    """fidelity p-values without the bootstrap floor (1/NB = 0.001 makes Holm over 100 edges unreachable): two-sided normal
    p from the bootstrap interval (s.e. = CI width / 3.92), Holm over the 2 * NFID edges; then the figure"""
    from math import erfc, sqrt
    res = json.load(open(FN)); fd = res['fidelity']
    se = [(c[1] - c[0]) / 3.92 for c in fd['actual_ci']]
    pn = [float(erfc(abs(a) / se_ / sqrt(2))) if se_ > 0 else 1.0 for a, se_ in zip(fd['actual'], se)]
    ph = holm(pn); n = len(pn) // 2
    fd['p_normal'] = pn; fd['p_normal_holm'] = ph
    fd['n_top_holm_sig_normal'] = int(sum(p < 0.05 for p in ph[:n])); fd['n_rand_holm_sig_normal'] = int(sum(p < 0.05 for p in ph[n:]))
    fd['note'] = 'p / p_holm use the bootstrap floor 1/NB and cannot pass Holm over 100 edges; use p_normal_holm'
    json.dump(res, open(FN, 'w'), indent=1)
    log(f"fidelity Holm (normal p): top {fd['n_top_holm_sig_normal']} / {n}, random {fd['n_rand_holm_sig_normal']} / {n}")
    plot(res)


if __name__ == '__main__':
    if '--post' in sys.argv: post()
    elif '--plot' in sys.argv: plot()
    else: main()
