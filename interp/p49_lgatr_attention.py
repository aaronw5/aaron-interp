"""P49 (loop C, round C11, exploratory): what L-GATr's attention heads compute. Do the attention logits of our L-GATr (run
anrunw-ai-c-p5-lgatr2, authors' jctagging config, 12 blocks x 8 heads, 2M random-file JetClass jets) reduce to a fixed function
of pair invariants, the way ParT's attention is mostly a fixed hinge formula of (ln kT, ln z, ln dR, ln m2) (claim C1)?

Terms. Multivector: element of the 16-dim space-time algebra; grade 0 = scalar (component 0), grade 1 = four-vector (t, x, y, z;
1-4), grade 2 = bivector (5-10), grade 3 = axial vector (11-14), grade 4 = pseudoscalar (15). L-GATr attention, per head h:
query / key of token i = 4 multivector channels (4 x 16) + 8 auxiliary scalar channels; logit
    a_ij = [ sum_c <q_i^c, k_j^c> + sum_s q_i^s k_j^s ] / sqrt(72),
with <.,.> the algebra's invariant inner product, factors (+1 | +1 -1 -1 -1 | -1 -1 -1 +1 +1 +1 | +1 +1 +1 -1 | -1). Its grade-1
part is the Minkowski product q_t k_t - q_x k_x - q_y k_y - q_z k_z. We split a_ij = A0 (grade 0) + A1 (grade 1, Minkowski) +
A234 (grades 2-4) + AS (auxiliary scalars). Tokens of a jet: position 0 global token (read out), 1 beam spurion (x-y plane
bivector), 2 time spurion (1,0,0,0), 3.. particles in pT order.

Row-centred: softmax over keys j is unchanged by adding a constant per query row i, so every logit, part and formula is compared
after subtracting its mean over the particle keys j of that row (particle query rows, particle keys = "pp block").

Sample (main / held-out): JetClass val_5M <class>_122.root (not among L-GATr's training or validation files; the planned _123 is
L-GATr's validation file for QCD, Hcc and Tbqq), 1,000 random rows per class, default_rng(4949), N 10,000, shuffled once.
Fit split: same files, 500 random rows per class from the remaining rows, default_rng(4950), N 5,000.

Steps
  base     unedited accuracy and 10-class confusion; plumbing: re-implemented attention (dense per jet) vs the model's own
           (xformers) forward; P1 sample (rows 3000-3300 of each _100 test file, N 3,000).
  fit      on the fit split: per head (96), row-centred pp-block logits regressed on pair features
             inv  = ln m2_ij, ln p_i.p_j                          (Lorentz invariants)
             invE = inv + ln(E_i E_j), ln E_j                     (frame-dependent energies added)
             part = ln kT, ln z, ln dR, ln m2                       (ParT's pair features, C1)
           as (i) additive hinge formula (per feature: x and 12 ReLU(x - knot) at quantiles; + diagonal indicator; row-centred
           least squares) and (ii) a per-head MLP (inputs -> 16 -> 16 -> 1, GELU), trained on row-centred targets.
           Null: the hinge fit with the keys of every row permuted (pair structure destroyed, row statistics kept).
  eval     on the main sample: per head, variance shares of A0/A1/A234/AS (Cov(part, total)/Var(total), row-centred, pp block and
           global-token row), R2 of the grade-1 part alone, held-out R2 of every formula with a 95 % bootstrap over jets.
  scan     each of the 96 heads in turn: uniform attention (all logits of that head equal -> uniform over the jet's tokens) and
           resample (the head's logit pattern taken from the pT-neighbour jet of the batch, particle k <- donor particle k mod n);
           accuracy change with bootstrap CI, McNemar on per-jet correctness, Holm over 96 heads per mode; the Holm-significant set S
           (uniform) ablated jointly vs 30 random head sets of |S|.
  circuit  every head's attention replaced at once:
             F1 (pp block): particle-query rows, particle keys <- formula (row-centred) + the live row mean; other logits live.
                            floor F1-0: formula = 0 (uniform over particles, same share to non-particle keys).
             F2 (pure):     particle-query rows: particle keys <- formula, global/spurion keys <- per-head constants fitted on the
                            fit split; global and spurion query rows uniform. Floor: uniform everywhere.
             F2g:           F2 with the global-token (read-out) query row kept live. Floor F2g-0: F2g with formula and constants 0.
           accuracy, confusion, same-call fraction and McNemar vs unedited, for inv / invE / part x hinge / MLP.
  untrained the same eval and circuit on a randomly initialised L-GATr (torch seed 0, same config).

  ./sync.sh && ./part_job.sh c-p49-run '/j-jepa-vol/aaron-interp/env-lgatr/bin/python -m interp.p49_lgatr_attention'
  (--smoke: 20 rows per class; --sections to restrict)
-> $AI_RESULTS/lgatr/p49_lgatr_attention.json, p49_arrays.npz, p49_lgatr_attention.png
"""
import argparse, glob, json, os, sys, time, traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('JETDISTILL_TAGGER', 'part')
ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--ckpt', default=None)
ap.add_argument('--sections', default='base,fit,eval,scan,circuit,untrained'); ap.add_argument('--figure_only', action='store_true')
ARGS = ap.parse_args(); SMOKE = ARGS.smoke
SEED, SEED_FIT, FIDX = 4949, 4950, 122
PER = 20 if SMOKE else 1000
PER_FIT = 10 if SMOKE else 500
NMLP = 60 if SMOKE else 2000
MLP_STEPS = 40 if SMOKE else 3000
NRSET = 3 if SMOKE else 30
NB = 100 if SMOKE else 1000
BS = 125
NBLK, NH = 12, 8
NHEAD = NBLK * NH
VAL = Path('/j-jepa-vol/JetClass/Pythia/val_5M')
PRE = {'QCD': 'ZJetsToNuNu', 'Hbb': 'HToBB', 'Hcc': 'HToCC', 'Hgg': 'HToGG', 'H4q': 'HToWW4Q', 'Hqql': 'HToWW2Q1L', 'Zqq': 'ZToQQ',
       'Wqq': 'WToQQ', 'Tbqq': 'TTBar', 'Tbl': 'TTBarLep'}
CL = ['QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']
iW, iZ = CL.index('Wqq'), CL.index('Zqq')
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'lgatr'
TAG = '_smoke' if SMOKE else ''
RUN = '/j-jepa-vol/aaron-interp/models/lgatr/runs/jc2m/jc2m_s0a/models'
IPF = [1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1]
FEATS = ['ln m2', 'ln p.p', 'ln EiEj', 'ln Ej', 'ln kT', 'ln z', 'ln dR']
FSETS = {'inv': [0, 1], 'invE': [0, 1, 2, 3], 'part': [4, 5, 6, 0]}
NKNOT = 12
PARTS = ['g0', 'g1', 'g234', 's']
T0 = time.time()
CTX = {}


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def jdump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(obj, indent=1, default=float)); log('wrote', path)


# ------------------------------------------------------------------------------------------------ sample
def read_sample(seed, per, exclude=None):
    import uproot
    from interp.part_core import read_root, sub, cat
    rng = np.random.default_rng(seed); parts = []; rows_all = []
    for c in CL:
        path = VAL / f'{PRE[c]}_{FIDX}.root'; N = uproot.open(path)['tree'].num_entries
        pool = np.arange(N)
        if exclude is not None: pool = np.setdiff1d(pool, exclude[exclude[:, 0] == CL.index(c), 1])
        rows = np.sort(rng.choice(pool, per, replace=False)); ch = []
        for s in range(0, N, 25000):
            r = rows[(rows >= s) & (rows < s + 25000)]
            if len(r): ch.append(sub(read_root(path, s, min(s + 25000, N)), r - s))
        J = cat(ch); assert (J['label'] == CL.index(c)).all(), c
        parts.append(J); rows_all.append(np.stack([np.full(len(rows), CL.index(c)), rows], 1))
    J = cat(parts); rows = np.concatenate(rows_all); perm = rng.permutation(len(rows))
    return sub(J, perm), rows[perm]


def latest_ckpt():
    fin = sorted(p for p in glob.glob(f'{RUN}/*.pt') if '_it' not in Path(p).name)
    if fin: return fin[-1]
    c = sorted(glob.glob(f'{RUN}/*_it*.pt'), key=lambda p: int(p.split('_it')[-1].split('.')[0])); return c[-1]


# ------------------------------------------------------------------------------------------------ pair features
def pair_tensors(J, sl, npm, dev):
    """padded (B, npm, npm, 7) pair features in FEATS order (0 on the diagonal), diag (B,npm,npm) bool, pv valid pairs"""
    import torch
    t = lambda k: torch.from_numpy(np.ascontiguousarray(J[k][sl][:, :npm])).to(dev).double()
    E, px, py, pz = t('energy'), t('px'), t('py'), t('pz'); v = torch.from_numpy(J['mask'][sl][:, :npm] > 0).to(dev)
    pt = torch.hypot(px, py).clamp(min=1e-9); Ec = E.clamp(min=1e-9)
    rap = 0.5 * torch.log((Ec + pz).clamp(min=1e-12) / (Ec - pz).clamp(min=1e-12)); phi = torch.atan2(py, px)
    a = lambda x: x[:, :, None]; b = lambda x: x[:, None, :]
    pdot = a(E) * b(E) - a(px) * b(px) - a(py) * b(py) - a(pz) * b(pz)
    msq = lambda x: x ** 2
    m2 = msq(a(E) + b(E)) - msq(a(px) + b(px)) - msq(a(py) + b(py)) - msq(a(pz) + b(pz))
    dphi = torch.remainder(a(phi) - b(phi) + np.pi, 2 * np.pi) - np.pi; dy = a(rap) - b(rap); dR = torch.sqrt(dy ** 2 + dphi ** 2)
    ptmin = torch.minimum(a(pt), b(pt)); kT = ptmin * dR; z = ptmin / (a(pt) + b(pt))
    L = lambda x: torch.log(x.clamp(min=1e-8))
    X = torch.stack([L(m2), L(pdot), L(a(Ec) * b(Ec)), L(b(Ec)).expand_as(m2), L(kT), L(z), L(dR)], -1)
    n = X.shape[1]; diag = torch.eye(n, dtype=torch.bool, device=dev)[None].expand(X.shape[0], n, n)
    pv = a(v) & b(v); X = X * (~diag & pv)[..., None]
    return X.float(), diag, pv


def basis(Xv, dv, fs, knots):
    import torch
    cols = []
    off = (~dv).float()[:, None]
    for f in fs:
        x = Xv[:, f:f + 1]; cols.append(x * off); cols.append(torch.relu(x - knots[f][None]) * off)
    cols.append(dv.float()[:, None])
    return torch.cat(cols, 1)


def seg_center(vals, rid, nrow):
    """subtract the per-row mean (rows given by rid) from vals (P, ...)"""
    import torch
    s = torch.zeros((nrow,) + vals.shape[1:], device=vals.device, dtype=vals.dtype).index_add_(0, rid, vals)
    c = torch.zeros(nrow, device=vals.device, dtype=vals.dtype).index_add_(0, rid, torch.ones(len(rid), device=vals.device, dtype=vals.dtype))
    c = c.clamp(min=1).reshape((nrow,) + (1,) * (vals.dim() - 1))
    return vals - (s / c)[rid]


# ------------------------------------------------------------------------------------------------ MLP (per head, vectorised)
def mlp_init(nh, fin, dev, hid=16, seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    p = dict(W1=r(nh, fin, hid) * fin ** -0.5, b1=torch.zeros(nh, hid), W2=r(nh, hid, hid) * hid ** -0.5, b2=torch.zeros(nh, hid),
             W3=r(nh, hid) * hid ** -0.5, b3=torch.zeros(nh))
    return {k: v.to(dev).requires_grad_(True) for k, v in p.items()}


def mlp_fwd(p, X, heads=None):
    import torch
    q = p if heads is None else {k: v[heads] for k, v in p.items()}
    g = torch.nn.functional.gelu
    h = g(torch.einsum('pf,hfk->hpk', X, q['W1']) + q['b1'][:, None])
    h = g(torch.einsum('hpk,hkl->hpl', h, q['W2']) + q['b2'][:, None])
    return torch.einsum('hpk,hk->hp', h, q['W3']) + q['b3'][:, None]


def mlp_in(Xv, dv, fs, mu, sd):
    import torch
    return torch.cat([(Xv[:, fs] - mu[fs]) / sd[fs] * (~dv).float()[:, None], dv.float()[:, None]], 1)


# ------------------------------------------------------------------------------------------------ model plumbing
class LGA:
    def __init__(self, ckpt, dev):
        import torch
        from interp.p5_lgatr_launch import load_lgatr, LG as SRC
        sys.path.insert(0, SRC)
        self.model, self.cfg = load_lgatr(ckpt, dev); self.dev = dev; self.ckpt = ckpt or 'random init (torch seed 0)'
        for p in self.model.parameters(): p.requires_grad_(False)
        from experiments.tagging.embedding import dense_to_sparse_jet, embed_tagging_data_into_ga
        self._d2s, self._emb = dense_to_sparse_jet, embed_tagging_data_into_ga
        self.ipf = torch.tensor(IPF, dtype=torch.float32, device=dev)
        for b, blk in enumerate(self.model.net.blocks):
            ga = blk.attention.attention; ga._orig_forward = ga.forward; ga.forward = self._make_att(b, ga._orig_forward)
        self.fit = None

    # ---- patched attention
    def _make_att(self, b, orig):
        import torch
        def att(q_mv, k_mv, v_mv, q_s, k_s, v_s, attention_mask=None):
            if CTX.get('orig'): return orig(q_mv, k_mv, v_mv, q_s, k_s, v_s, attention_mask=attention_mask)
            G, V = CTX['G'], CTX['V']; H, T, C = q_mv.shape[1], q_mv.shape[2], q_mv.shape[-2]; Cv = v_mv.shape[-2]
            B, n = G.shape; d = C * 16 + q_s.shape[-1]; sc = d ** -0.5
            Q = q_mv[0][:, G] * self.ipf; K = k_mv[0][:, G]; Qs = q_s[0][:, G]; Ks = k_s[0][:, G]
            if CTX.get('parts'):
                prt = []
                for xs in ([0], [1, 2, 3, 4], list(range(5, 16))):
                    prt.append(Q[..., xs].reshape(H, B, n, -1) @ K[..., xs].reshape(H, B, n, -1).transpose(-1, -2) * sc)
                prt.append(Qs @ Ks.transpose(-1, -2) * sc)
                L = prt[0] + prt[1] + prt[2] + prt[3]
                CTX['cap_parts'][b] = prt
            else:
                L = (Q.reshape(H, B, n, C * 16) @ K.reshape(H, B, n, C * 16).transpose(-1, -2) + Qs @ Ks.transpose(-1, -2)) * sc
            if CTX.get('capture'): CTX['cap'][b] = L
            spec = CTX.get('edit', {}).get(b)
            if spec: L = self._edit(b, L, spec)
            L = L.masked_fill(~V[None, :, None, :], float('-inf'))
            A = torch.softmax(L, -1)
            Vc = torch.cat([v_mv[0][:, G].reshape(H, B, n, Cv * 16), v_s[0][:, G]], -1)
            O = A @ Vc
            out = O.new_zeros(H, T, O.shape[-1]); out[:, G[V]] = O[:, V]
            return out[..., :Cv * 16].reshape(1, H, T, Cv, 16), out[..., Cv * 16:].unsqueeze(0)
        return att

    def _edit(self, b, L, spec):
        import torch
        L = L.clone()
        if spec.get('uniform'): L[spec['uniform']] = 0.0
        if spec.get('resample'):
            hs = spec['resample']; dn, M = CTX['donor'], CTX['M']
            L[hs] = L[hs][:, dn[:, None, None], M[:, :, None], M[:, None, :]]
        if spec.get('grow'): L[:, :, 0, :] = 0.0
        if spec.get('srow'): L[:, :, 1:3, :] = 0.0
        if spec.get('formula'):
            fam, fs, kind = spec['formula']; bi, ii, jj = CTX['pidx']; rid = CTX['rid']; nrow = CTX['nrow']
            hsl = slice(b * NH, (b + 1) * NH)
            if fs is None:
                pc = torch.zeros(NH, len(bi), device=L.device)
            else:
                pred = self.predict(fs, kind, hsl)                     # (8, P)
                pc = seg_center(pred.T.contiguous(), rid, nrow).T
            if fam == 'F1c':                                            # particle rows pure formula + fitted constants; rows 0-2 live
                L[:, bi, ii + 3, jj + 3] = pc
                L[:, :, 3:, 0:3] = self.fit['const'][hsl][:, None, None, :]
            elif fam == 'F1':
                live = L[:, bi, ii + 3, jj + 3]
                rm = live.T.contiguous(); rm = rm - seg_center(rm, rid, nrow)   # live row mean, per pair
                L[:, bi, ii + 3, jj + 3] = pc + rm.T
            else:                                                       # F2: pure
                L2 = torch.zeros_like(L)
                L2[:, bi, ii + 3, jj + 3] = pc
                cst = self.fit['const'][hsl] if fs is not None else torch.zeros(NH, 3, device=L.device)
                L2[:, :, 3:, 0:3] = cst[:, None, None, :]
                if fam == 'F2g': L2[:, :, 0, :] = L[:, :, 0, :]          # read-out (global-token) row kept live
                L = L2
        return L

    def predict(self, fs, kind, hsl):
        """formula prediction (uncentred) for the current batch's valid particle pairs, heads hsl -> (len, P)"""
        F = self.fit; Xv, dv = CTX['Xv'], CTX['dv']
        if kind == 'hinge':
            key = ('phi', fs)
            if key not in CTX: CTX[key] = basis(Xv, dv, FSETS[fs], F['knots'])
            return (CTX[key] @ F['W'][fs][:, hsl]).T
        key = ('min', fs)
        if key not in CTX: CTX[key] = mlp_in(Xv, dv, FSETS[fs], F['mu'], F['sd'])
        heads = list(range(hsl.start, hsl.stop))
        out = []
        for s in range(0, CTX[key].shape[0], 60000):
            out.append(mlp_fwd(F['mlp'][fs], CTX[key][s:s + 60000], heads))
        import torch
        return torch.cat(out, 1)

    # ---- embedding with context
    def embed(self, J, sl):
        import torch
        from interp.p5_lgatr_launch import to_lgatr_inputs
        from interp.part_core import sub
        Js = sub(J, sl); vec, feat = to_lgatr_inputs(Js)
        fm, sc, ptr = self._d2s(torch.from_numpy(vec).to(self.dev), torch.from_numpy(feat).to(self.dev))
        e = self._emb(fm, sc, ptr, self.cfg.data)
        bt = e['batch']; nb = int(bt.max()) + 1; cnt = torch.bincount(bt, minlength=nb); start = torch.cumsum(cnt, 0) - cnt
        pos = torch.arange(len(bt), device=bt.device) - start[bt]
        assert torch.equal(e['is_global'], pos == 0), 'global token not first'
        npart = cnt - 3; nreal = torch.from_numpy((Js['mask'] > 0).sum(1)).to(self.dev)
        assert torch.equal(npart, nreal), 'token count != particle count + 3'
        n = int(cnt.max()); ar = torch.arange(n, device=self.dev)
        V = ar[None] < cnt[:, None]; G = torch.where(V, start[:, None] + ar[None], torch.zeros_like(V, dtype=torch.long))
        # donor (pT neighbour) for resample
        jpt = torch.from_numpy(Js['jet_pt'].astype(np.float64)).to(self.dev); o = torch.argsort(jpt); rk = torch.empty_like(o); rk[o] = torch.arange(nb, device=self.dev)
        nbr = torch.where(rk + 1 < nb, rk + 1, rk - 1).clamp(min=0); donor = o[nbr]
        M = torch.where(ar[None] < 3, ar[None].expand(nb, n), 3 + (ar[None] - 3).clamp(min=0) % npart[donor][:, None].clamp(min=1))
        X, diag, pv = pair_tensors(Js, slice(None), n - 3, self.dev)
        bi, ii, jj = pv.nonzero(as_tuple=True); rid = bi * (n - 3) + ii
        for k in [k for k in CTX if isinstance(k, tuple)]: del CTX[k]
        CTX.update(G=G, V=V, donor=donor, M=M, Xv=X[pv], dv=diag[pv], pidx=(bi, ii, jj), rid=rid, nrow=nb * (n - 3), nb=nb, npm=n - 3,
                   cap={}, cap_parts={})
        e['nb'] = nb
        return e

    def forward(self, e, k0=0, ins=None, cache=False):
        from interp.p5_lgatr_launch import TO_CL
        m = self.model; net = m.net
        if ins is None: h_mv, h_s = net.linear_in(e['mv'].unsqueeze(0), scalars=e['s'].unsqueeze(0))
        else: h_mv, h_s = ins
        cached = []
        for j in range(k0, NBLK):
            if cache: cached.append((h_mv, h_s))
            h_mv, h_s = net.blocks[j](h_mv, scalars=h_s, attention_mask=None)
        o_mv, o_s = net.linear_out(h_mv, scalars=h_s)
        return m.extract_from_ga(o_mv, o_s, e['batch'], e['is_global'])[:, TO_CL], cached

    def logits(self, J, edit=None):
        import torch
        N = len(J['label']); out = np.zeros((N, 10), np.float32)
        with torch.no_grad():
            for s in range(0, N, BS):
                sl = np.arange(s, min(s + BS, N)); e = self.embed(J, sl); CTX['edit'] = edit or {}
                out[sl] = self.forward(e)[0].float().cpu().numpy()
        CTX['edit'] = {}
        return out


# ------------------------------------------------------------------------------------------------ statistics
def boot_mean(x, nb=NB, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x, float); n = len(x); idx = rng.integers(0, n, (nb, n))
    bs = x[idx].mean(1); return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def mcnemar(c0, c1):
    from scipy.stats import binomtest
    b, c = int((c0 & ~c1).sum()), int((~c0 & c1).sum())
    return dict(lost=b, gained=c, p=float(binomtest(b, b + c).pvalue) if b + c else 1.0)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.zeros(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


def confusion(y, p):
    M = np.zeros((10, 10), int); np.add.at(M, (y, p), 1); return M


def compare(L0, L1, y, seed=0):
    p0, p1 = L0.argmax(1), L1.argmax(1); W, Z = y == iW, y == iZ; c0, c1 = p0 == y, p1 == y
    return dict(acc=boot_mean(c1.astype(float), seed=seed), dacc=boot_mean(c1.astype(float) - c0.astype(float), seed=seed),
                same_call=float((p0 == p1).mean()), mcnemar=mcnemar(c0, c1),
                W_to_Z=[float((p0[W] == iZ).mean()), float((p1[W] == iZ).mean())], Z_to_W=[float((p0[Z] == iW).mean()), float((p1[Z] == iW).mean())],
                per_class_recall={c: float((p1[y == i] == i).mean()) for i, c in enumerate(CL)}, confusion=confusion(y, p1).tolist(),
                mean_abs_dlogit=float(np.abs(L1 - L0).mean()), logit_sd_unedited=float((L0 - L0.mean(1, keepdims=True)).std()))


def r2_boot(sse, sst, nb=NB, seed=0):
    """sse, sst: (N, H) per-jet sums -> (H, 3) R2 with 95 % bootstrap interval over jets"""
    rng = np.random.default_rng(seed); n = len(sse); r = 1 - sse.sum(0) / np.maximum(sst.sum(0), 1e-12); bs = []
    for _ in range(nb):
        i = rng.integers(0, n, n); bs.append(1 - sse[i].sum(0) / np.maximum(sst[i].sum(0), 1e-12))
    bs = np.array(bs); return np.stack([r, np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)], 1)


# ------------------------------------------------------------------------------------------------ sections
def sec_base(lg, Jb, y, J1, out, arr, name):
    import torch
    L0 = lg.logits(Jb)
    with torch.no_grad():
        sl = np.arange(0, min(BS, len(y))); e = lg.embed(Jb, sl); CTX['orig'] = True
        try:
            Lw = lg.model({k: e[k] for k in ('mv', 's', 'is_global', 'batch')})
        finally:
            CTX['orig'] = False
        from interp.p5_lgatr_launch import TO_CL
        Lw = Lw[:, TO_CL].float().cpu().numpy(); dsplit = float(np.abs(Lw - L0[sl]).max()); same = float((Lw.argmax(1) == L0[sl].argmax(1)).mean())
        # embedding check: particle four-vectors in the tokens vs the jet arrays (units of 20 GeV)
        mvp = e['mv'][:, 0, 1:5][(CTX['G'][:, 3:][CTX['V'][:, 3:]])].cpu().numpy()
        from interp.part_core import sub
        Js = sub(Jb, sl); m = Js['mask'] > 0; ref = np.stack([Js['energy'][m], Js['px'][m], Js['py'][m], Js['pz'][m]], 1) / 20.0
        emb_dev = float(np.abs(mvp - ref).max() / np.abs(ref).max())
    p0 = L0.argmax(1); r = dict(N=int(len(y)), acc=boot_mean((p0 == y).astype(float)), confusion=confusion(y, p0).tolist(),
                                per_class_recall={c: float((p0[y == i] == i).mean()) for i, c in enumerate(CL)},
                                patched_vs_model_forward_max_abs_dlogit=dsplit, patched_vs_model_same_call=same, embed_particle_order_rel_dev=emb_dev)
    if J1 is not None:
        L1 = lg.logits(J1); r['P1_sample'] = dict(N=int(len(J1['label'])), acc=float((L1.argmax(1) == J1['label'].astype(int)).mean()))
    out[name] = r; arr[f'{name}_L0'] = L0
    log(f"{name}: acc {100 * r['acc'][0]:.2f} [{100 * r['acc'][1]:.2f}, {100 * r['acc'][2]:.2f}]; patched vs model forward max|dL| {dsplit:.3e} "
        f"same call {same:.4f}; embedding order dev {emb_dev:.2e}; P1 sample {r.get('P1_sample')}")
    return L0


def pp_targets():
    """row-centred pp logits for all 96 heads at the current batch: (P, 96), plus per-head live row means and key-type offsets"""
    import torch
    bi, ii, jj = CTX['pidx']; rid, nrow = CTX['rid'], CTX['nrow']
    Y = torch.cat([CTX['cap'][k][:, bi, ii + 3, jj + 3] for k in range(NBLK)], 0).T.contiguous()   # (P, 96)
    return seg_center(Y, rid, nrow), Y


def sec_fit(lg, Jf, out):
    """normal equations for the hinge formulas (+ permuted-key null), F2 constants, MLP training data; then solve and train"""
    import torch
    dev = lg.dev; N = len(Jf['label']); K = {}; A = {}; Bm = {}; Bn = {}; csum = torch.zeros(NHEAD, 3, dtype=torch.float64, device=dev); crow = 0
    store = dict(X=[], d=[], Y=[], rid=[], jet=[]); Xs = []; rowbase = 0; g = torch.Generator(device='cpu').manual_seed(SEED_FIT)
    with torch.no_grad():
        for s in range(0, N, BS):
            sl = np.arange(s, min(s + BS, N)); e = lg.embed(Jf, sl); CTX['capture'] = True; CTX['edit'] = {}
            lg.forward(e); CTX['capture'] = False
            Yc, Yraw = pp_targets(); Xv, dv = CTX['Xv'], CTX['dv']; bi, ii, jj = CTX['pidx']; rid, nrow = CTX['rid'], CTX['nrow']
            if s == 0:
                off = Xv[~dv]; qs = torch.linspace(0, 1, NKNOT + 2, device=dev)[1:-1]
                K['knots'] = {f: torch.quantile(off[:200000, f], qs) for f in range(len(FEATS))}
                K['mu'] = off.mean(0); K['sd'] = off.std(0) + 1e-6
            # key-type offsets: logit(i, key t) - live pp row mean, particle rows
            rm = torch.zeros(nrow, NHEAD, device=dev, dtype=torch.float64).index_add_(0, rid, Yraw.double())
            cn = torch.zeros(nrow, device=dev, dtype=torch.float64).index_add_(0, rid, torch.ones(len(rid), device=dev, dtype=torch.float64))
            okr = cn > 0; rm = rm[okr] / cn[okr][:, None]
            rb, ri = torch.div(torch.arange(nrow, device=dev), CTX['npm'], rounding_mode='floor')[okr], torch.remainder(torch.arange(nrow, device=dev), CTX['npm'])[okr]
            for t in range(3):
                Lt = torch.cat([CTX['cap'][k][:, rb, ri + 3, t] for k in range(NBLK)], 0).T.double()   # (rows, 96)
                csum[:, t] += (Lt - rm).sum(0)
            crow += int(okr.sum())
            # permuted-key null: within each row, keys permuted among the row's particles
            npj = (CTX['V'][:, 3:]).sum(1)
            rnd = torch.rand(CTX['nb'], CTX['npm'], CTX['npm'], generator=g).to(dev)
            rnd = rnd.masked_fill(~CTX['V'][:, None, 3:], 2.0); order = torch.argsort(rnd, -1)
            Ypad = torch.zeros(CTX['nb'], CTX['npm'], CTX['npm'], NHEAD, device=dev); Ypad[bi, ii, jj] = Yc
            Yn = Ypad[bi, ii, order[bi, ii, jj]]
            for fs in FSETS:
                Phi = seg_center(basis(Xv, dv, FSETS[fs], K['knots']), rid, nrow).double()
                A[fs] = A.get(fs, 0) + Phi.T @ Phi; Bm[fs] = Bm.get(fs, 0) + Phi.T @ Yc.double(); Bn[fs] = Bn.get(fs, 0) + Phi.T @ Yn.double()
            if s < NMLP:
                store['X'].append(Xv.half()); store['d'].append(dv); store['Y'].append(Yc.half()); store['rid'].append(rid + rowbase)
                store['jet'].append(bi + s)
            rowbase += nrow
            log(f'fit batch {s // BS + 1}/{(N + BS - 1) // BS}: pairs {len(bi)}')
    W, Wn = {}, {}
    for fs in FSETS:
        lam = 1e-6 * torch.diagonal(A[fs]).mean(); I = torch.eye(A[fs].shape[0], device=dev, dtype=torch.float64)
        W[fs] = torch.linalg.solve(A[fs] + lam * I, Bm[fs]).float(); Wn[fs] = torch.linalg.solve(A[fs] + lam * I, Bn[fs]).float()
    K.update(W=W, Wnull=Wn, const=(csum / max(crow, 1)).float())
    lg.fit = K
    # MLP training
    X = torch.cat(store['X']); D = torch.cat(store['d']); Y = torch.cat(store['Y']); R = torch.cat(store['rid']); Jt = torch.cat(store['jet'])
    jets = torch.unique(Jt); ends = torch.searchsorted(Jt, jets, right=True); starts = torch.searchsorted(Jt, jets, right=False)
    var = Y.float().var(0) + 1e-6
    K['mlp'] = {}; K['mlp_loss'] = {}
    for fs in FSETS:
        p = mlp_init(NHEAD, len(FSETS[fs]) + 1, dev, seed={'inv': 1, 'invE': 2, 'part': 3}[fs]); opt = torch.optim.Adam(list(p.values()), lr=3e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MLP_STEPS); losses = []
        gg = torch.Generator(device='cpu').manual_seed(SEED_FIT + 1)
        for step in range(MLP_STEPS):
            pick = torch.randint(0, len(jets), (16,), generator=gg)
            ix = torch.cat([torch.arange(int(starts[q]), int(ends[q]), device=dev) for q in pick])
            rid_u, rinv = torch.unique(R[ix], return_inverse=True)
            xin = mlp_in(X[ix].float(), D[ix], FSETS[fs], K['mu'], K['sd'])
            pr = seg_center(mlp_fwd(p, xin).T.contiguous(), rinv, len(rid_u))
            loss = (((pr - Y[ix].float()) ** 2).mean(0) / var).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step(); losses.append(float(loss))
        for v in p.values(): v.requires_grad_(False)
        K['mlp'][fs] = p; K['mlp_loss'][fs] = [float(np.mean(losses[:20])), float(np.mean(losses[-50:]))]
        log(f'MLP {fs}: normalised loss first/last {K["mlp_loss"][fs]}')
    out['fit'] = dict(N_fit=int(N), N_mlp_jets=int(len(jets)), mlp_steps=MLP_STEPS, mlp_loss=K['mlp_loss'],
                      knots={FEATS[f]: v.cpu().tolist() for f, v in K['knots'].items()},
                      F2_constants_global_beam_time=K['const'].cpu().tolist())
    del store, X, Y, R; torch.cuda.empty_cache()


def sec_eval(lg, Jb, out, arr, name):
    """variance decomposition and held-out R2 on the main sample"""
    import torch
    dev = lg.dev; N = len(Jb['label']); F = lg.fit
    preds = [f'{fs}_{k}' for fs in FSETS for k in ('hinge', 'mlp', 'null')] + ['g1_only']
    sse = {p: np.zeros((N, NHEAD)) for p in preds}; sst = np.zeros((N, NHEAD))
    dec = {w: dict(TT=np.zeros(NHEAD), PT=np.zeros((4, NHEAD)), PP=np.zeros((4, NHEAD))) for w in ('pp', 'global')}
    with torch.no_grad():
        for s in range(0, N, BS):
            sl = np.arange(s, min(s + BS, N)); e = lg.embed(Jb, sl); CTX.update(capture=True, parts=True, edit={})
            lg.forward(e); CTX.update(capture=False, parts=False)
            bi, ii, jj = CTX['pidx']; rid, nrow = CTX['rid'], CTX['nrow']; Xv, dv = CTX['Xv'], CTX['dv']
            Yc, _ = pp_targets()
            Pc = [seg_center(torch.cat([CTX['cap_parts'][k][g][:, bi, ii + 3, jj + 3] for k in range(NBLK)], 0).T.contiguous(), rid, nrow) for g in range(4)]
            dec['pp']['TT'] += (Yc.double() ** 2).sum(0).cpu().numpy()
            for g in range(4):
                dec['pp']['PT'][g] += (Pc[g].double() * Yc.double()).sum(0).cpu().numpy(); dec['pp']['PP'][g] += (Pc[g].double() ** 2).sum(0).cpu().numpy()
            # global-token row: query position 0, particle keys
            gv = CTX['V'][:, 3:]; gb, gj = gv.nonzero(as_tuple=True)
            Yg = seg_center(torch.cat([CTX['cap'][k][:, gb, 0, gj + 3] for k in range(NBLK)], 0).T.contiguous(), gb, CTX['nb'])
            dec['global']['TT'] += (Yg.double() ** 2).sum(0).cpu().numpy()
            for g in range(4):
                Pg = seg_center(torch.cat([CTX['cap_parts'][k][g][:, gb, 0, gj + 3] for k in range(NBLK)], 0).T.contiguous(), gb, CTX['nb'])
                dec['global']['PT'][g] += (Pg.double() * Yg.double()).sum(0).cpu().numpy(); dec['global']['PP'][g] += (Pg.double() ** 2).sum(0).cpu().numpy()
            jet = torch.from_numpy(sl).to(dev)[bi]
            def acc_sse(pc, key):
                v = torch.zeros(N, NHEAD, device=dev, dtype=torch.float64).index_add_(0, jet, ((pc - Yc) ** 2).double())
                sse[key][sl] += v[sl].cpu().numpy()
            sst[sl] += torch.zeros(N, NHEAD, device=dev, dtype=torch.float64).index_add_(0, jet, (Yc ** 2).double())[sl].cpu().numpy()
            for fs in FSETS:
                Phi = basis(Xv, dv, FSETS[fs], F['knots'])
                acc_sse(seg_center(Phi @ F['W'][fs], rid, nrow), f'{fs}_hinge')
                acc_sse(seg_center(Phi @ F['Wnull'][fs], rid, nrow), f'{fs}_null')
                xin = mlp_in(Xv, dv, FSETS[fs], F['mu'], F['sd'])
                pr = torch.cat([mlp_fwd(F['mlp'][fs], xin[q:q + 30000]) for q in range(0, len(xin), 30000)], 1).T.contiguous()
                acc_sse(seg_center(pr, rid, nrow), f'{fs}_mlp')
            acc_sse(Pc[1], 'g1_only')
            if (s // BS) % 10 == 0: log(f'{name} eval batch {s // BS + 1}/{(N + BS - 1) // BS}')
    R2 = {p: r2_boot(sse[p], sst, seed=7) for p in preds}
    res = dict(R2={p: v.tolist() for p, v in R2.items()}, heads='index = block * 8 + head', parts=PARTS)
    for w in ('pp', 'global'):
        D = dec[w]; res[f'share_{w}'] = (D['PT'] / np.maximum(D['TT'], 1e-12)).tolist(); res[f'varfrac_{w}'] = (D['PP'] / np.maximum(D['TT'], 1e-12)).tolist()
    summ = {}
    for p in preds:
        r = R2[p][:, 0]; summ[p] = dict(median=float(np.median(r)), frac_ge_0p7=float((r >= 0.7).mean()), frac_lt_0p3=float((r < 0.3).mean()),
                                        per_block_median=[float(np.median(r[b * NH:(b + 1) * NH])) for b in range(NBLK)])
    res['summary'] = summ
    sh = np.array(res['share_pp'])
    res['share_pp_median_per_part'] = {PARTS[g]: float(np.median(sh[g])) for g in range(4)}
    out[name] = res; arr[f'{name}_sse'] = np.stack([sse[p] for p in preds]).astype(np.float32); arr[f'{name}_sst'] = sst.astype(np.float32)
    log(f'{name}: median share pp ' + ' '.join(f'{PARTS[g]} {np.median(sh[g]):.3f}' for g in range(4)))
    for p in preds: log(f"{name}: R2 {p:12s} median {summ[p]['median']:.3f} frac>=0.7 {summ[p]['frac_ge_0p7']:.2f} frac<0.3 {summ[p]['frac_lt_0p3']:.2f} "
                        f"per block {' '.join(f'{v:.2f}' for v in summ[p]['per_block_median'])}")


def sec_scan(lg, Jb, y, L0, out, arr):
    import torch
    N = len(y); P = np.zeros((2, NHEAD, N), np.int8)
    with torch.no_grad():
        for s in range(0, N, BS):
            sl = np.arange(s, min(s + BS, N)); e = lg.embed(Jb, sl); CTX['edit'] = {}
            _, ins = lg.forward(e, cache=True)
            for b in range(NBLK):
                for h in range(NH):
                    for mi, mode in enumerate(('uniform', 'resample')):
                        CTX['edit'] = {b: {mode: [h]}}
                        P[mi, b * NH + h, sl] = lg.forward(e, k0=b, ins=ins[b])[0].argmax(1).cpu().numpy()
            CTX['edit'] = {}
            if (s // BS) % 8 == 0: log(f'scan batch {s // BS + 1}/{(N + BS - 1) // BS}')
    arr['scan_pred'] = P
    p0 = L0.argmax(1); c0 = p0 == y; res = {}
    rng = np.random.default_rng(11); idx = rng.integers(0, N, (NB, N))
    for mi, mode in enumerate(('uniform', 'resample')):
        rows = []
        for hd in range(NHEAD):
            c1 = P[mi, hd] == y; d = c1.astype(float) - c0.astype(float); bs = d[idx].mean(1); W = y == iW
            rows.append(dict(head=f'b{hd // NH}h{hd % NH}', acc=float(c1.mean()), dacc=[float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                             calls_changed=float((P[mi, hd] != p0).mean()), mcnemar=mcnemar(c0, c1),
                             W_to_Z=[float((p0[W] == iZ).mean()), float((P[mi, hd][W] == iZ).mean())]))
        ph = holm([r['mcnemar']['p'] for r in rows])
        for r, p in zip(rows, ph): r['p_holm'] = float(p)
        res[mode] = rows
        sig = [r['head'] for r in rows if r['p_holm'] < 0.05]
        top = sorted(rows, key=lambda r: r['dacc'][0])[:5]
        log(f"scan {mode}: Holm-significant heads {len(sig)}: {sig[:20]}; largest drops " + ', '.join(f"{r['head']} {100 * r['dacc'][0]:+.2f}" for r in top))
    out['scan'] = res
    # joint ablation of the significant set vs random sets of the same size
    rows = res['uniform']; S = [i for i, r in enumerate(rows) if r['p_holm'] < 0.05 and r['dacc'][0] < 0]
    S = sorted(S, key=lambda i: rows[i]['dacc'][0])[:16]
    if not S: S = sorted(range(NHEAD), key=lambda i: rows[i]['dacc'][0])[:5]
    def spec(hs):
        d = {}
        for hd in hs: d.setdefault(hd // NH, {'uniform': []})['uniform'].append(hd % NH)
        return d
    Ls = lg.logits(Jb, spec(S)); accS = float((Ls.argmax(1) == y).mean())
    rng = np.random.default_rng(12); racc = []
    for _ in range(NRSET):
        hs = rng.choice(NHEAD, len(S), replace=False); racc.append(float((lg.logits(Jb, spec(hs)).argmax(1) == y).mean()))
    out['scan_set'] = dict(set=[f'b{i // NH}h{i % NH}' for i in S], size=len(S), acc_set=boot_mean((Ls.argmax(1) == y).astype(float)), acc_random=racc,
                           acc_random_mean=float(np.mean(racc)), acc_random_2p5_97p5=[float(np.percentile(racc, 2.5)), float(np.percentile(racc, 97.5))],
                           rank_p=float((1 + sum(a <= accS for a in racc)) / (1 + len(racc))), mcnemar_vs_unedited=mcnemar(c0, Ls.argmax(1) == y))
    log(f"scan set {out['scan_set']['set']}: acc {100 * accS:.2f} vs random sets {100 * np.mean(racc):.2f} "
        f"[{100 * np.percentile(racc, 2.5):.2f}, {100 * np.percentile(racc, 97.5):.2f}] rank p {out['scan_set']['rank_p']:.3f}")


def sec_circuit(lg, Jb, y, L0, out, arr, name, kinds=('hinge', 'mlp')):
    allb = lambda sp: {b: dict(sp) for b in range(NBLK)}
    runs = {'uniform_all': allb({'uniform': list(range(NH))}), 'global_row_uniform': allb({'grow': True}),
            'F1_floor': allb({'formula': ('F1', None, None)}), 'F2g_floor': allb({'formula': ('F2g', None, None)})}
    for fam in ('F1', 'F2', 'F2g'):
        for fs in FSETS:
            for k in kinds: runs[f'{fam}_{fs}_{k}'] = allb({'formula': (fam, fs, k)})
    res = {}; acc0 = float((L0.argmax(1) == y).mean())
    for rn, sp in runs.items():
        L1 = lg.logits(Jb, sp); r = compare(L0, L1, y); res[rn] = r; arr[f'{name}_{rn}_pred'] = L1.argmax(1).astype(np.int8)
        log(f"{name} {rn:20s} acc {100 * r['acc'][0]:.2f} [{100 * r['acc'][1]:.2f}, {100 * r['acc'][2]:.2f}] (unedited {100 * acc0:.2f}) same call {r['same_call']:.3f} "
            f"McNemar p {r['mcnemar']['p']:.2e} W->Z {r['W_to_Z'][0]:.3f}->{r['W_to_Z'][1]:.3f}")
    for fam, floor in (('F1', 'F1_floor'), ('F2', 'uniform_all'), ('F2g', 'F2g_floor')):
        fa = res[floor]['acc'][0]
        for rn in res:
            if rn.startswith(fam + '_') and rn != floor:
                res[rn]['recovered_fraction'] = float((res[rn]['acc'][0] - fa) / max(acc0 - fa, 1e-9))
    out[name] = dict(unedited_acc=acc0, runs=res)


def sec_circuit2(lg, Jb, y, L0, out, arr, name='circuit2', kinds=('hinge', 'mlp')):
    """follow-up: where the gap between F1 and F2g arises. F1c = particle query rows entirely from the formula (particle keys) and
    the fitted per-head constants (global / spurion keys), global and spurion query rows live; srow = only the two spurion query
    rows uniform."""
    allb = lambda sp: {b: dict(sp) for b in range(NBLK)}
    runs = {'spurion_rows_uniform': allb({'srow': True}), 'F1c_floor': allb({'formula': ('F1c', None, None)})}
    for fs in FSETS:
        for k in kinds:
            runs[f'F1c_{fs}_{k}'] = allb({'formula': ('F1c', fs, k)}); runs[f'F1_{fs}_{k}_srow'] = allb({'formula': ('F1', fs, k), 'srow': True})
    res = {}; acc0 = float((L0.argmax(1) == y).mean())
    for rn, sp in runs.items():
        L1 = lg.logits(Jb, sp); r = compare(L0, L1, y); res[rn] = r; arr[f'{name}_{rn}_pred'] = L1.argmax(1).astype(np.int8)
        log(f"{name} {rn:24s} acc {100 * r['acc'][0]:.2f} [{100 * r['acc'][1]:.2f}, {100 * r['acc'][2]:.2f}] (unedited {100 * acc0:.2f}) same call {r['same_call']:.3f} "
            f"McNemar p {r['mcnemar']['p']:.2e} W->Z {r['W_to_Z'][0]:.3f}->{r['W_to_Z'][1]:.3f}")
    out[name] = dict(unedited_acc=acc0, runs=res)


# ------------------------------------------------------------------------------------------------ figure
def figure(out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'text.usetex': False, 'font.family': 'DejaVu Sans'})
    ev = out['eval']; fig, ax = plt.subplots(1, 3, figsize=(18, 5.4), gridspec_kw=dict(width_ratios=[1.1, 1.2, 1.1]))
    sh = np.array(ev['share_pp']); x = np.arange(NBLK); cols = ['C7', 'C0', 'C2', 'C1']
    lab = {'g0': 'grade 0 (scalar)', 'g1': 'grade 1 (Minkowski product)', 'g234': 'grades 2-4', 's': 'auxiliary scalar channels'}
    bot = np.zeros(NBLK)
    for g in range(4):
        v = np.array([np.median(sh[g, b * NH:(b + 1) * NH]) for b in range(NBLK)]); ax[0].bar(x, v, bottom=bot, color=cols[g], label=lab[PARTS[g]]); bot += v
    ax[0].set_xlabel('block'); ax[0].set_ylabel('share of row-centred logit variance (median over 8 heads)')
    ax[0].set_title('(a) which part of q·k varies across particle pairs\n(particle query rows, particle keys)', fontsize=9); ax[0].legend(fontsize=7)
    R = {k: np.array(v)[:, 0] for k, v in ev['R2'].items()}
    for k, c, mk, nm in (('inv_mlp', 'C0', 'o', 'invariants ln m², ln p·p (MLP)'), ('invE_mlp', 'C3', 's', '+ ln E_iE_j, ln E_j (MLP)'),
                         ('part_hinge', 'C2', '^', "ParT features, hinge (C1 form)"), ('g1_only', 'C7', 'x', 'grade-1 part alone'),
                         ('part_null', 'k', '.', 'null: keys permuted in each row')):
        jit = {'inv_mlp': -0.24, 'invE_mlp': -0.08, 'part_hinge': 0.08, 'g1_only': 0.24, 'part_null': 0.0}[k]
        ax[1].scatter(np.repeat(x, NH) + jit, R[k], s=10, c=c, marker=mk, label=nm)
    if 'untrained_eval' in out:
        Ru = np.array(out['untrained_eval']['R2']['invE_mlp'])[:, 0]
        ax[1].plot(x, [np.median(Ru[b * NH:(b + 1) * NH]) for b in range(NBLK)], 'k:', label='untrained, + ln E (MLP), median')
    ax[1].axhline(0.7, color='grey', ls='--', lw=0.7); ax[1].axhline(0.3, color='grey', ls=':', lw=0.7)
    ax[1].set_ylim(-0.2, 1.02); ax[1].set_xlabel('block'); ax[1].set_ylabel('held-out R² of row-centred logits (10,000 jets)')
    ax[1].set_title('(b) how much of each head is a fixed function of pair features\n(one point per head)', fontsize=9); ax[1].legend(fontsize=6.5, loc='lower left')
    C = out['circuit']['runs']; a0 = 100 * out['circuit']['unedited_acc']
    names = [('uniform_all', 'uniform attention everywhere'), ('F2_inv_mlp', 'pure formula: invariants'), ('F2_invE_mlp', 'pure formula: + ln E'),
             ('F2_part_hinge', 'pure formula: ParT hinge'), ('F1_floor', 'pp block uniform'), ('F1_inv_mlp', 'pp block: invariants'),
             ('F1_invE_mlp', 'pp block: + ln E'), ('F1_part_hinge', 'pp block: ParT hinge'), ('F2g_floor', 'all uniform but read-out row'),
             ('F2g_inv_mlp', 'formula + live read-out: invariants'), ('F2g_invE_mlp', 'formula + live read-out: + ln E'),
             ('F2g_part_hinge', 'formula + live read-out: ParT hinge'), ('global_row_uniform', 'read-out row uniform only')]
    v = [100 * C[k]['acc'][0] for k, _ in names]; lo = [100 * C[k]['acc'][1] for k, _ in names]; hi = [100 * C[k]['acc'][2] for k, _ in names]
    yy = np.arange(len(names))
    ax[2].barh(yy, v, color=['C7'] + ['C0'] * 3 + ['C7'] + ['C2'] * 3 + ['C7'] + ['C3'] * 3 + ['C1'], xerr=[np.subtract(v, lo), np.subtract(hi, v)])
    ax[2].axvline(a0, color='k', ls='--', lw=0.8, label=f'unedited {a0:.1f} %'); ax[2].axvline(10, color='grey', ls=':', lw=0.8, label='chance 10 %')
    ax[2].set_yticks(yy); ax[2].set_yticklabels([n for _, n in names], fontsize=7.5); ax[2].set_xlabel('10-class accuracy [%] (10,000 jets)')
    ax[2].set_title('(c) every head\'s attention replaced by its fitted pair formula', fontsize=9); ax[2].legend(fontsize=7)
    fig.tight_layout(); p = RES / f'p49_lgatr_attention{TAG}.png'; fig.savefig(p, dpi=130); log('wrote', p)


# ------------------------------------------------------------------------------------------------ main
def main():
    global TAG
    if ARGS.sections == 'base,fit,circuit2': TAG = TAG + '_circuit2'
    out_path = RES / f'p49_lgatr_attention{TAG}.json'
    if ARGS.figure_only:
        figure(json.loads(out_path.read_text())); return
    import torch
    from interp.part_core import kin, build, sample
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; ck = ARGS.ckpt or latest_ckpt(); log('checkpoint', ck, 'device', dev)
    secs = ARGS.sections.split(',')
    J, rows = read_sample(SEED, PER); y = J['label'].astype(int); Jb = build(J, *kin(J))
    Jf0, rows_f = read_sample(SEED_FIT, PER_FIT, exclude=rows); Jf = build(Jf0, *kin(Jf0))
    assert not set(map(tuple, rows)) & set(map(tuple, rows_f)), 'fit rows overlap main rows'
    J1 = sample(30 if SMOKE else 300, 3000); J1 = build(J1, *kin(J1))
    log('samples: main', len(y), 'fit', len(Jf['label']), 'P1', len(J1['label']))
    out = dict(node='P49', ckpt=ck, sample=f'val_5M <class>_{FIDX}.root, {PER} random rows per class, default_rng({SEED}), shuffled',
               fit_sample=f'same files, {PER_FIT} other rows per class, default_rng({SEED_FIT})', N=int(len(y)), feats=FEATS, fsets=FSETS,
               heads='head index = block * 8 + head; b<block>h<head>, 0-based')
    arr = dict(rows=rows, rows_fit=rows_f, y=y); RES.mkdir(parents=True, exist_ok=True)
    lg = LGA(ck, dev); log('loaded L-GATr; force_xformers', lg.model.force_xformers)
    def step(nm, fn):
        try: fn()
        except Exception:
            log(f'SECTION {nm} FAILED'); traceback.print_exc(); out.setdefault('failed', []).append(nm)
        jdump(out, out_path); np.savez_compressed(RES / f'p49_arrays{TAG}.npz', **arr)
    L0 = {}
    step('base', lambda: L0.__setitem__('t', sec_base(lg, Jb, y, J1, out, arr, 'base')))
    if 'fit' in secs: step('fit', lambda: sec_fit(lg, Jf, out))
    if 'eval' in secs and lg.fit: step('eval', lambda: sec_eval(lg, Jb, out, arr, 'eval'))
    if 'circuit' in secs and lg.fit: step('circuit', lambda: sec_circuit(lg, Jb, y, L0['t'], out, arr, 'circuit'))
    if 'circuit2' in secs and lg.fit: step('circuit2', lambda: sec_circuit2(lg, Jb, y, L0['t'], out, arr))
    if 'scan' in secs: step('scan', lambda: sec_scan(lg, Jb, y, L0['t'], out, arr))
    if 'untrained' in secs:
        del lg; torch.cuda.empty_cache()
        lu = LGA(None, dev); log('loaded untrained L-GATr')
        step('untrained_base', lambda: L0.__setitem__('u', sec_base(lu, Jb, y, None, out, arr, 'untrained_base')))
        step('untrained_fit', lambda: sec_fit(lu, Jf, out.setdefault('untrained', {})))
        if lu.fit:
            step('untrained_eval', lambda: sec_eval(lu, Jb, out, arr, 'untrained_eval'))
            step('untrained_circuit', lambda: sec_circuit(lu, Jb, y, L0['u'], out, arr, 'untrained_circuit'))
    if 'eval' in out:
        try:
            figure(out)
        except Exception:
            traceback.print_exc()
    log('done')


if __name__ == '__main__':
    main()
