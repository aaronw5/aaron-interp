"""P55 (loop C, round C12, exploratory; claim C26): literal attention-head circuits of our L-GATr for W vs Z and t vs QCD.

Question. Does a small set of L-GATr attention heads, kept alone while every other head is resample-ablated, reproduce most of
L-GATr's binary W-vs-Z and t-vs-QCD accuracy, and do those heads sit in blocks 0-4 (where P49 found the 21 heads whose single
removal costs accuracy)? Reference: ParT_full's top-quark head circuit (P37), quoted as-is.

Model. L-GATr final checkpoint (runs/jc2m/jc2m_s0a/models/model_run0.pt; authors' jctagging config; 12 blocks x 8 heads = 96
heads; 83.73 % on the P1 sample). Control model: the same architecture with random weights (torch seed 0, as P49).

Head output. In block b, head h produces O_bh = softmax(q k^T / sqrt(72)) V for every token (dimension 72 = 4 multivector channels x
16 components + 8 auxiliary scalars), before the per-head scale and the output projection. All edits act on O_bh:
  resample ablation: O_bh(jet) <- O_bh(donor) computed on the unedited donor jet with the unedited network; donor = a jet of the
    donor pool chosen at random among the 5 jets nearest in jet pT (any class); token mapping: global token and the two spurion
    tokens to themselves, particle k (pT order) <- donor particle k mod n_donor.
  mean ablation (secondary): O_bh <- its mean over the reference set (200 jets per class, 10 classes), separately for the global
    token, the beam spurion, the time spurion and the particle tokens.
  A configuration is a weight w_bh in [0, 1] per head (or per head and output component): O = O_donor + w (O_live - O_donor);
  w = 1 kept, w = 0 ablated.

Samples (JetClass test_20M; files not used before in loop C and not in L-GATr's training (train_100M) or validation (val_5M)).
  ranking: <class>_119.root, default_rng(5555): 2,000 random rows per class for W, Z, t->bqq', QCD; plus a reference set of 200
           other random rows per class for all 10 classes (donor pool for ranking, and the mean for mean ablation).
           greedy subset: 400 of the 2,000 per class (random).
  evaluation: <class>_113.root, default_rng(5556): 1,000 random rows per class, N 10,000; donors from the same sample (not self).
  P1 sample (rows 3,000-3,300 of each _100 file, N 3,000) for the unedited-accuracy check.
Read-outs. Task WZ: d = l_W - l_Z on true W and Z jets; task tQCD: d = l_t - l_QCD on true t->bqq' and QCD jets. Binary accuracy =
fraction of the task's jets with sign(d) correct. Metric for ranking: M = mean over the task's jets of s_y d (s_y = +1 for the first
class, -1 for the second).

Steps
  base     unedited accuracy (10-class, binary) on the evaluation and P1 samples; patched vs model forward; w = 1 vs no edit.
  means    per-head mean outputs on the reference set.
  rank     on the ranking sample, per task: (i) greedy removal from the full model on the greedy subset (each live head removed in
           turn, the heads whose removal leaves M highest are removed, 16 at a time down to 32 live heads, then 4 at a time; the
           ranking = reverse removal order; primary); (ii) EAP-IG: integrated gradients of M with respect to the per-head
           weights w along the straight path from all heads ablated (w = 0) to all kept (w = 1), 16 midpoint steps; score =
           mean gradient (completeness: sum of scores ~ M(1) - M(0)); secondary. Rankings written before evaluation.
  circuits on the evaluation sample, per task, ranking and k in 4, 8, 16, 24, 32: keep the top-k heads, ablate the other 96 - k.
           10-class accuracy and confusion, binary accuracy, faithfulness F = (acc_circuit - acc_floor) / (acc_unedited -
           acc_floor) with acc_floor = all 96 heads ablated; logit-difference faithfulness F_d the same with M; 95 % bootstrap
           over jets. Mean-ablation variant for the greedy circuits.
  null     per task and k on the task's evaluation jets: 100 block-matched random k-sets (same number of heads per block as the
           greedy circuit), 30 block-matched to the EAP-IG circuit, 50 unrestricted random k-sets. Rank p = (1 + #{random acc >=
           circuit acc}) / (1 + n_random); Holm across the 10 (task, k) tests per comparison family.
  roles    per task: every head removed alone from the full model and kept alone (all other 95 ablated), binary accuracy and
           McNemar (Holm over 96); for the heads of the greedy k = 16 circuit: circuit minus that head, and the circuit with only
           that head's grade-0 / grade-1 / grade-2-4 multivector components or its auxiliary scalars resample-ablated; the share of
           the head's output sum of squares in each part (all tokens; global token).
  e4       task WZ: matched real-mass dilation E4 (every particle's (deta, dphi) about the stored axis x 1.1; jet and pair masses rise,
           mean dlog m reported): R_c = sum_jets [d_c(E4) - d_c(base)] / sum_jets [d_full(E4) - d_full(base)] for the greedy
           circuits (donor outputs from the unedited donors), the all-ablated floor and 20 block-matched random sets per k.
  untrained random-weight L-GATr: unedited, floor, EAP-IG-ranked circuits (binary and 10-class accuracy).

  ./sync.sh && ./part_job.sh c-p55-run '/j-jepa-vol/aaron-interp/env-lgatr/bin/python -m interp.p55_lgatr_head_circuit'
  (--smoke: small sample; --figure_only)
-> $AI_RESULTS/lgatr/p55_lgatr_head_circuit.json, p55_ranking.json, p55_arrays.npz, p55_lgatr_head_circuit.png
"""
import argparse, json, os, sys, time, traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('JETDISTILL_TAGGER', 'part')
ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--figure_only', action='store_true')
ap.add_argument('--ckpt', default='/j-jepa-vol/aaron-interp/models/lgatr/runs/jc2m/jc2m_s0a/models/model_run0.pt')
ap.add_argument('--sections', default='base,means,rank,circuits,null,roles,e4,untrained')
ARGS = ap.parse_args(); SMOKE = ARGS.smoke
RANK_FILE, EVAL_FILE = 119, 113
SEED_R, SEED_E = 5555, 5556
PER_RANK = 40 if SMOKE else 2000
PER_REF = 10 if SMOKE else 200
PER_GREEDY = 20 if SMOKE else 400
PER_EVAL = 30 if SMOKE else 1000
KS = [4, 8, 16, 24, 32]
N_BM, N_BM_EAP, N_UN, N_E4 = (4, 2, 3, 2) if SMOKE else (100, 30, 50, 20)
IG_STEPS = 16
NB = 200 if SMOKE else 1000
BS = 128
TOK_BUDGET = 1.6e7
TOK_MAX = 24000          # tokens per forward (the MLP geometric-product einsum needs ~65 kB per token; 94k tokens ran out of 11 GB)
CMAX = 32
K_DONOR = 5
DIL = 1.1
NBLK, NH = 12, 8
NHEAD = NBLK * NH
TEST = Path('/j-jepa-vol/JetClass/Pythia/test_20M')
PRE = {'QCD': 'ZJetsToNuNu', 'Hbb': 'HToBB', 'Hcc': 'HToCC', 'Hgg': 'HToGG', 'H4q': 'HToWW4Q', 'Hqql': 'HToWW2Q1L', 'Zqq': 'ZToQQ',
       'Wqq': 'WToQQ', 'Tbqq': 'TTBar', 'Tbl': 'TTBarLep'}
CL = ['QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']
TASKS = {'WZ': (CL.index('Wqq'), CL.index('Zqq')), 'tQCD': (CL.index('Tbqq'), CL.index('QCD'))}
TCLS = ['Wqq', 'Zqq', 'Tbqq', 'QCD']
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'lgatr'
TAG = '_smoke' if SMOKE else ''
IPF = [1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1]
GRADES = {'g0': [0], 'g1': [1, 2, 3, 4], 'g234': list(range(5, 16))}
T0 = time.time()
CTX = {}
hn = lambda i: f'b{i // NH}h{i % NH}'


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def jdump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(obj, indent=1, default=float)); log('wrote', path)


# ------------------------------------------------------------------------------------------------ data
def nent(c, fidx):
    import uproot
    return uproot.open(TEST / f'{PRE[c]}_{fidx}.root')['tree'].num_entries


def read_rows(c, fidx, rows):
    from interp.part_core import read_root, sub, cat
    path = TEST / f'{PRE[c]}_{fidx}.root'; N = nent(c, fidx); rows = np.sort(rows); ch = []
    for s in range(0, N, 20000):
        r = rows[(rows >= s) & (rows < s + 20000)]
        if len(r): ch.append(sub(read_root(path, s, min(s + 20000, N)), r - s))
    J = cat(ch); assert (J['label'] == CL.index(c)).all(), c
    return J


def load_samples():
    from interp.part_core import sub, cat, kin, build
    rng = np.random.default_rng(SEED_R); R, F = [], []; rrows, frows = [], []
    for c in CL:
        N = nent(c, RANK_FILE); perm = rng.permutation(N)
        rk = perm[:PER_RANK] if c in TCLS else perm[:0]; rf = perm[len(rk):len(rk) + PER_REF]
        J = read_rows(c, RANK_FILE, np.concatenate([rk, rf])); allr = np.sort(np.concatenate([rk, rf]))
        isr = np.isin(allr, rk)
        if isr.any(): R.append(sub(J, np.where(isr)[0])); rrows.append(np.stack([np.full(isr.sum(), CL.index(c)), allr[isr]], 1))
        F.append(sub(J, np.where(~isr)[0])); frows.append(np.stack([np.full((~isr).sum(), CL.index(c)), allr[~isr]], 1))
        log(f'ranking file {c}: {len(rk)} ranking + {len(rf)} reference rows')
    Jr, Jf = cat(R), cat(F); rr, fr = np.concatenate(rrows), np.concatenate(frows)
    p = rng.permutation(len(rr)); Jr, rr = sub(Jr, p), rr[p]
    yr = Jr['label'].astype(int); greedy = np.zeros(len(yr), bool)
    for c in TCLS:
        ix = np.where(yr == CL.index(c))[0]; greedy[rng.choice(ix, PER_GREEDY, replace=False)] = True
    rng = np.random.default_rng(SEED_E); E, erows = [], []
    for c in CL:
        N = nent(c, EVAL_FILE); rows = rng.choice(N, PER_EVAL, replace=False); E.append(read_rows(c, EVAL_FILE, rows))
        erows.append(np.stack([np.full(len(rows), CL.index(c)), np.sort(rows)], 1)); log(f'evaluation file {c}: {len(rows)} rows')
    Je, er = cat(E), np.concatenate(erows); p = rng.permutation(len(er)); Je, er = sub(Je, p), er[p]
    b = lambda J: build(J, *kin(J))
    return b(Jr), rr, greedy, b(Jf), fr, b(Je), er


def pick_donors(pt_t, pt_pool, same, seed):
    rng = np.random.default_rng(seed); lp = np.log(np.maximum(pt_pool, 1e-3)); o = np.argsort(lp); ls = lp[o]
    out = np.empty(len(pt_t), int)
    for i, x in enumerate(np.log(np.maximum(pt_t, 1e-3))):
        p = np.searchsorted(ls, x); cand = o[max(0, p - K_DONOR - 2):p + K_DONOR + 2]
        if same: cand = cand[cand != i]
        cand = cand[np.argsort(np.abs(lp[cand] - x))[:K_DONOR]]; out[i] = rng.choice(cand)
    return out


# ------------------------------------------------------------------------------------------------ model
class LGA:
    def __init__(self, ckpt, dev):
        import torch
        from interp.p5_lgatr_launch import load_lgatr, LG as SRC
        sys.path.insert(0, SRC)
        self.model, self.cfg = load_lgatr(ckpt, dev); self.dev = dev
        for p in self.model.parameters(): p.requires_grad_(False)
        from experiments.tagging.embedding import dense_to_sparse_jet, embed_tagging_data_into_ga
        self._d2s, self._emb = dense_to_sparse_jet, embed_tagging_data_into_ga
        self.ipf = torch.tensor(IPF, dtype=torch.float32, device=dev)
        for b, blk in enumerate(self.model.net.blocks):
            ga = blk.attention.attention; ga.forward = self._make_att(b, ga.forward)
        self.means = None

    def _make_att(self, b, orig):
        import torch
        def att(q_mv, k_mv, v_mv, q_s, k_s, v_s, attention_mask=None):
            if CTX.get('orig'): return orig(q_mv, k_mv, v_mv, q_s, k_s, v_s, attention_mask=attention_mask)
            G, V = CTX['G'], CTX['V']; H, T, C = q_mv.shape[1], q_mv.shape[2], q_mv.shape[-2]; Cv = v_mv.shape[-2]
            B, n = G.shape; sc = (C * 16 + q_s.shape[-1]) ** -0.5
            Q = q_mv[0][:, G] * self.ipf; K = k_mv[0][:, G]
            L = (Q.reshape(H, B, n, C * 16) @ K.reshape(H, B, n, C * 16).transpose(-1, -2) + q_s[0][:, G] @ k_s[0][:, G].transpose(-1, -2)) * sc
            L = L.masked_fill(~V[None, :, None, :], float('-inf'))
            A = torch.softmax(L, -1)
            O = A @ torch.cat([v_mv[0][:, G].reshape(H, B, n, Cv * 16), v_s[0][:, G]], -1)       # (H, B, n, 72)
            if CTX.get('capture') is not None: CTX['capture'][b] = O.detach()
            W = CTX.get('W')
            if W is not None:
                R = CTX['R'][b]; ji = CTX.get('jidx')
                Rr = R if ji is None else R[:, ji]
                O = W[b] * O + (1 - W[b]) * Rr                      # exact O for w = 1, exact Rr for w = 0
            out = O.new_zeros(H, T, O.shape[-1]); out[:, G[V]] = O[:, V]
            return out[..., :Cv * 16].reshape(1, H, T, Cv, 16), out[..., Cv * 16:].unsqueeze(0)
        return att

    def embed(self, Js):
        import torch
        from interp.p5_lgatr_launch import to_lgatr_inputs
        vec, feat = to_lgatr_inputs(Js)
        fm, sc, ptr = self._d2s(torch.from_numpy(vec).to(self.dev), torch.from_numpy(feat).to(self.dev))
        e = self._emb(fm, sc, ptr, self.cfg.data)
        bt = e['batch']; nb = int(bt.max()) + 1; cnt = torch.bincount(bt, minlength=nb); start = torch.cumsum(cnt, 0) - cnt
        pos = torch.arange(len(bt), device=bt.device) - start[bt]
        assert torch.equal(e['is_global'], pos == 0), 'global token not first'
        npart = cnt - 3; assert torch.equal(npart, torch.from_numpy((Js['mask'] > 0).sum(1)).to(self.dev)), 'token count'
        n = int(cnt.max()); ar = torch.arange(n, device=self.dev)
        V = ar[None] < cnt[:, None]; G = torch.where(V, start[:, None] + ar[None], torch.zeros_like(V, dtype=torch.long))
        return dict(mv=e['mv'], s=e['s'], batch=bt, is_global=e['is_global'], G=G, V=V, nb=nb, n=n, npart=npart, T=len(bt))

    @staticmethod
    def rep(e, C):
        import torch
        if C == 1: return e
        T, nb = e['T'], e['nb']; off = torch.arange(C, device=e['G'].device)
        return dict(mv=e['mv'].repeat(C, 1, 1), s=e['s'].repeat(C, 1), batch=(e['batch'][None] + off[:, None] * nb).reshape(-1),
                    is_global=e['is_global'].repeat(C), G=(e['G'][None] + off[:, None, None] * T).reshape(-1, e['n']),
                    V=e['V'].repeat(C, 1), nb=nb * C, n=e['n'], T=T * C)

    def forward(self, e, W=None, R=None, jidx=None, capture=None):
        from interp.p5_lgatr_launch import TO_CL
        m = self.model; net = m.net
        CTX.update(G=e['G'], V=e['V'], W=W, R=R, jidx=jidx, capture=capture)
        try:
            h_mv, h_s = net.linear_in(e['mv'].unsqueeze(0), scalars=e['s'].unsqueeze(0))
            for blk in net.blocks: h_mv, h_s = blk(h_mv, scalars=h_s, attention_mask=None)
            o_mv, o_s = net.linear_out(h_mv, scalars=h_s)
            return m.extract_from_ga(o_mv, o_s, e['batch'], e['is_global'])[:, TO_CL]
        finally:
            CTX.update(W=None, R=None, jidx=None, capture=None)


def make_batches(lg, J, idx, donors, Jpool):
    """target jets J[idx] sorted by multiplicity in batches of BS; donor jets Jpool[donors[idx]] with the token map"""
    import torch
    from interp.part_core import sub
    idx = np.asarray(idx); n = (J['mask'][idx] > 0).sum(1); idx = idx[np.argsort(n, kind='stable')]; out = []
    with torch.no_grad():
        for s in range(0, len(idx), BS):
            ii = idx[s:s + BS]; et = lg.embed(sub(J, ii)); ed = lg.embed(sub(Jpool, donors[ii]))
            ar = torch.arange(et['n'], device=lg.dev)
            Md = torch.where(ar[None] < 3, ar[None].expand(et['nb'], -1), 3 + (ar[None] - 3).clamp(min=0) % ed['npart'][:, None].clamp(min=1))
            out.append(dict(idx=ii, et=et, ed=ed, Md=Md.clamp(max=ed['n'] - 1)))
    return out


def donor_R(lg, bt, mode):
    import torch
    with torch.no_grad():
        if mode == 'resample':
            cap = {}; lg.forward(bt['ed'], capture=cap); ai = torch.arange(bt['et']['nb'], device=lg.dev)[:, None]
            return [cap[b][:, ai, bt['Md']] for b in range(NBLK)]
        typ = torch.arange(bt['et']['n'], device=lg.dev).clamp(max=3)
        return [lg.means[b][:, typ][:, None].expand(-1, bt['et']['nb'], -1, -1) for b in range(NBLK)]


def eval_configs(lg, batches, cfgs, N, mode='resample', e_key='et'):
    """cfgs (C, 96, g) weights -> logits (C, N, 10) numpy (rows = jet index of the sample; jets not in batches stay nan)"""
    import torch
    cf = torch.as_tensor(np.asarray(cfgs, np.float32), device=lg.dev); Cn = len(cf)
    out = np.full((Cn, N, 10), np.nan, np.float32)
    with torch.no_grad():
        for bt in batches:
            R = donor_R(lg, bt, mode); et = bt[e_key]; B, n = et['nb'], et['n']
            cm = int(max(1, min(CMAX, TOK_BUDGET // (B * n * n), TOK_MAX // et['T']))); reps = {}
            for c0 in range(0, Cn, cm):
                cc = cf[c0:c0 + cm]; C = len(cc)
                if C not in reps: reps[C] = LGA.rep(et, C)
                W = [cc[:, b * NH:(b + 1) * NH].permute(1, 0, 2).repeat_interleave(B, 1)[:, :, None, :] for b in range(NBLK)]
                L = lg.forward(reps[C], W=W, R=R, jidx=torch.arange(C * B, device=lg.dev) % B)
                out[c0:c0 + C, bt['idx']] = L.reshape(C, B, 10).float().cpu().numpy()
            del R
    return out


def plain_logits(lg, batches, N, e_key='et'):
    import torch
    out = np.full((N, 10), np.nan, np.float32)
    with torch.no_grad():
        for bt in batches: out[bt['idx']] = lg.forward(bt[e_key]).float().cpu().numpy()
    return out


# ------------------------------------------------------------------------------------------------ statistics
def boot_ci(x, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed); i = rng.integers(0, len(x), (NB, len(x))); b = x[i].mean(1)
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


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


def task_mask(y, t):
    a, b = TASKS[t]; return (y == a) | (y == b)


def bin_correct(L, y, t):
    """L (..., N, 10) -> per-jet correctness on the task's jets (..., n_task)"""
    a, b = TASKS[t]; m = task_mask(y, t)
    return ((L[..., a] > L[..., b]) == (y == a))[..., m]


def sdiff(L, y, t):
    a, b = TASKS[t]; m = task_mask(y, t); s = np.where(y[m] == a, 1.0, -1.0)
    return s * (L[..., a] - L[..., b])[..., m]


def faith(cc, cf, cu, seed=0):
    """F = (mean cc - mean cf) / (mean cu - mean cf) with bootstrap over jets (per-jet arrays, same jets)"""
    rng = np.random.default_rng(seed); n = len(cc); i = rng.integers(0, n, (NB, n))
    f = lambda a, b_, c: (a.mean(-1) - b_.mean(-1)) / np.where(np.abs(c.mean(-1) - b_.mean(-1)) > 1e-9, c.mean(-1) - b_.mean(-1), np.nan)
    pt = float(f(cc.astype(float), cf.astype(float), cu.astype(float))); bs = f(cc[i].astype(float), cf[i].astype(float), cu[i].astype(float))
    return [pt, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]


def keep_cfg(heads):
    w = np.zeros((NHEAD, 1), np.float32); w[list(heads)] = 1; return w


def bm_random(chosen, n, rng):
    hist = np.bincount(np.asarray(chosen) // NH, minlength=NBLK); out = []
    for _ in range(n):
        out.append(np.concatenate([b * NH + rng.choice(NH, hist[b], replace=False) for b in range(NBLK) if hist[b]]))
    return out


# ------------------------------------------------------------------------------------------------ sections
def sec_base(lg, S, out, arr):
    import torch
    from interp.part_core import kin, build, sample
    Je, ye = S['Je'], S['ye']; L0 = plain_logits(lg, S['Be'], len(ye)); p0 = L0.argmax(1)
    bt = S['Be'][len(S['Be']) // 2]
    with torch.no_grad():
        CTX['orig'] = True
        try:
            Lm = lg.model({k: bt['et'][k] for k in ('mv', 's', 'is_global', 'batch')})
        finally:
            CTX['orig'] = False
        from interp.p5_lgatr_launch import TO_CL
        Lm = Lm[:, TO_CL].float().cpu().numpy()
    L1 = eval_configs(lg, S['Be'][:2], np.ones((1, NHEAD, 1)), len(ye))[0]; ok = ~np.isnan(L1[:, 0])
    r = dict(N=int(len(ye)), acc10=boot_ci(p0 == ye), confusion=confusion(ye, p0).tolist(),
             recall={c: float((p0[ye == i] == i).mean()) for i, c in enumerate(CL)},
             patched_vs_model_max_abs_dlogit=float(np.abs(Lm - L0[bt['idx']]).max()), patched_vs_model_same_call=float((Lm.argmax(1) == p0[bt['idx']]).mean()),
             w1_vs_noedit_max_abs_dlogit=float(np.abs(L1[ok] - L0[ok]).max()), w1_vs_noedit_same_call=float((L1[ok].argmax(1) == L0[ok].argmax(1)).mean()))
    for t in TASKS:
        r[f'bin_{t}'] = boot_ci(bin_correct(L0, ye, t)); r[f'M_{t}'] = float(sdiff(L0, ye, t).mean())
    if True:
        J1 = sample(30 if SMOKE else 300, 3000); J1 = build(J1, *kin(J1)); y1 = J1['label'].astype(int)
        B1 = make_batches(lg, J1, np.arange(len(y1)), np.arange(len(y1)), J1)
        L1p = plain_logits(lg, B1, len(y1)); r['P1_sample'] = dict(N=int(len(y1)), acc10=float((L1p.argmax(1) == y1).mean()))
    out['base'] = r; arr['eval_L0'] = L0
    log(f"base: 10-class acc {100 * r['acc10'][0]:.2f} [{100 * r['acc10'][1]:.2f}, {100 * r['acc10'][2]:.2f}]; W/Z binary {100 * r['bin_WZ'][0]:.2f}; "
        f"t/QCD binary {100 * r['bin_tQCD'][0]:.2f}; patched vs model max|dL| {r['patched_vs_model_max_abs_dlogit']:.2e} same call "
        f"{r['patched_vs_model_same_call']:.4f}; w=1 vs no edit {r['w1_vs_noedit_max_abs_dlogit']:.2e}; P1 {r.get('P1_sample')}")
    return L0


def sec_means(lg, S, out):
    import torch
    sums = [None] * NBLK; cnt = torch.zeros(4, device=lg.dev)
    with torch.no_grad():
        for bt in S['Bf']:
            cap = {}; lg.forward(bt['et'], capture=cap); V = bt['et']['V']; n = bt['et']['n']
            typ = torch.arange(n, device=lg.dev).clamp(max=3); oh = torch.nn.functional.one_hot(typ, 4).float()   # (n, 4)
            Vf = V.float()
            for b in range(NBLK):
                s = torch.einsum('hbnd,bn,nt->htd', cap[b], Vf, oh); sums[b] = s if sums[b] is None else sums[b] + s
            cnt += torch.einsum('bn,nt->t', Vf, oh)
    lg.means = [s / cnt[None, :, None] for s in sums]
    out['means'] = dict(N_ref=int(len(S['yf'])), token_counts=cnt.cpu().tolist())
    log(f'means from {len(S["yf"])} reference jets; token counts {cnt.cpu().tolist()}')


def sec_rank(lg, S, out, arr, key='rank', greedy_on=True, write=True):
    import torch
    Jr, yr = S['Jr'], S['yr']; res = {}
    for t in TASKS:
        a, b = TASKS[t]; m = task_mask(yr, t); r = {}
        # endpoints
        Bt = S[f'Br_{t}']; Lend = eval_configs(lg, Bt, np.stack([np.ones((NHEAD, 1)), np.zeros((NHEAD, 1))]), len(yr))
        Mend = sdiff(Lend, yr, t).mean(-1); r['M_full_floor'] = Mend.tolist()
        r['bin_full_floor'] = bin_correct(Lend, yr, t).mean(-1).tolist()
        # EAP-IG
        g = np.zeros((IG_STEPS, NHEAD)); Mpath = np.zeros(IG_STEPS); ntask = int(m.sum())
        for bt in Bt:
            R = donor_R(lg, bt, 'resample'); yy = torch.from_numpy(yr[bt['idx']]).to(lg.dev)
            sgn = torch.where(yy == a, 1.0, torch.where(yy == b, -1.0, 0.0))
            for s in range(IG_STEPS):
                al = torch.full((NHEAD,), (s + 0.5) / IG_STEPS, device=lg.dev, requires_grad=True)
                with torch.enable_grad():
                    W = [al[k * NH:(k + 1) * NH].view(NH, 1, 1, 1) for k in range(NBLK)]
                    L = lg.forward(bt['et'], W=W, R=R)
                    Mb = (sgn * (L[:, a] - L[:, b])).sum() / ntask
                    Mb.backward()
                g[s] += al.grad.cpu().numpy(); Mpath[s] += float(Mb)
            del R
        ig = g.mean(0); r['eapig'] = ig.tolist(); r['eapig_path_M'] = dict(alpha=[(s + 0.5) / IG_STEPS for s in range(IG_STEPS)], M=Mpath.tolist()); r['eapig_completeness'] = dict(sum_scores=float(ig.sum()), M_full_minus_floor=float(Mend[0] - Mend[1]))
        r['eapig_order'] = [int(i) for i in np.argsort(-ig)]
        log(f"{key} {t}: M full {Mend[0]:.3f} floor {Mend[1]:.3f}; EAP-IG sum {ig.sum():.3f}; top-16 {[hn(i) for i in r['eapig_order'][:16]]}")
        # greedy removal
        if greedy_on:
            yg = np.where(S['greedy'], yr, -1)                         # only the greedy-subset jets count
            Bg = S[f'Bg_{t}']; live = np.ones(NHEAD, bool); removed = []; stages = []
            while live.any():
                cand = np.where(live)[0]; chunk = 16 if live.sum() > 32 else 4
                cf = np.repeat(live[None, :, None].astype(np.float32), len(cand), 0); cf[np.arange(len(cand)), cand, 0] = 0
                M = sdiff(eval_configs(lg, Bg, cf, len(yr)), yg, t).mean(-1)
                o = np.argsort(-M)[:chunk]; rem = cand[o]; live[rem] = False; removed += [int(h) for h in rem]
                stages.append(dict(n_live_before=int(len(cand)), removed=[hn(h) for h in rem], M_after_each=[float(M[i]) for i in o]))
                log(f"{key} {t} greedy: {len(cand)} live -> removed {[hn(h) for h in rem]} (M after {M[o[-1]]:.3f})")
            r['greedy_order'] = removed[::-1]; r['greedy_stages'] = stages
            # M of the kept top-k on the greedy subset
            Lk = eval_configs(lg, Bg, np.stack([keep_cfg(r['greedy_order'][:k]) for k in KS]), len(yr))
            r['greedy_rank_sample_M_topk'] = dict(zip(map(str, KS), sdiff(Lk, yg, t).mean(-1).tolist()))
            r['greedy_rank_sample_bin_topk'] = dict(zip(map(str, KS), bin_correct(Lk, yg, t).mean(-1).tolist()))
            log(f"{key} {t}: greedy top-16 {[hn(i) for i in r['greedy_order'][:16]]}; M of top-k on greedy subset {r['greedy_rank_sample_M_topk']}")
        res[t] = r
    out[key] = res
    if write:
        jdump(dict(node='P55', frozen_before_evaluation=True, ckpt=ARGS.ckpt,
                   rankings={t: dict(greedy=[hn(i) for i in res[t]['greedy_order']], eapig=[hn(i) for i in res[t]['eapig_order']]) for t in res}),
              RES / f'p55_ranking{TAG}.json')


def circuit_sets(out, key='rank'):
    sets = {}
    for t in TASKS:
        for rk in ('greedy', 'eapig'):
            o = out[key][t].get(f'{rk}_order')
            if o is None: continue
            for k in KS: sets[(t, rk, k)] = [int(i) for i in o[:k]]
    return sets


def sec_circuits(lg, S, out, arr, L0, key='circuits', rkey='rank', mean_variant=True):
    ye = S['ye']; sets = circuit_sets(out, rkey); names = list(sets)
    cfgs = [np.ones((NHEAD, 1)), np.zeros((NHEAD, 1))] + [keep_cfg(sets[s]) for s in names]
    L = eval_configs(lg, S['Be'], np.stack(cfgs), len(ye)); res = {}
    Lm = eval_configs(lg, S['Be'], np.stack([np.zeros((NHEAD, 1))] + [keep_cfg(sets[s]) for s in names if s[1] == 'greedy']), len(ye), mode='mean') if mean_variant else None
    def summ(Lc, Lf, Lu, t, extra=None):
        p = Lc.argmax(1); cc, cf, cu = bin_correct(Lc, ye, t), bin_correct(Lf, ye, t), bin_correct(Lu, ye, t)
        dc, df, du = sdiff(Lc, ye, t), sdiff(Lf, ye, t), sdiff(Lu, ye, t)
        r = dict(bin_acc=boot_ci(cc), F=faith(cc, cf, cu), F_logitdiff=faith(dc, df, du), acc10=boot_ci(p == ye),
                 confusion=confusion(ye, p).tolist(), recall={c: float((p[ye == i] == i).mean()) for i, c in enumerate(CL)},
                 mcnemar_vs_unedited=mcnemar(cu, cc))
        if extra: r.update(extra)
        return r
    res['unedited'] = {t: dict(bin_acc=boot_ci(bin_correct(L[0], ye, t))) for t in TASKS}
    res['floor_resample'] = dict(acc10=boot_ci(L[1].argmax(1) == ye), confusion=confusion(ye, L[1].argmax(1)).tolist(),
                                 **{f'bin_{t}': boot_ci(bin_correct(L[1], ye, t)) for t in TASKS})
    if Lm is not None:
        res['floor_mean'] = dict(acc10=boot_ci(Lm[0].argmax(1) == ye), **{f'bin_{t}': boot_ci(bin_correct(Lm[0], ye, t)) for t in TASKS})
    gi = 1
    for i, s in enumerate(names):
        t, rk, k = s; blocks = np.bincount(np.array(sets[s]) // NH, minlength=NBLK)
        r = summ(L[2 + i], L[1], L[0], t, dict(heads=[hn(h) for h in sets[s]], per_block=blocks.tolist(), frac_blocks_0_4=float(blocks[:5].sum() / k)))
        if Lm is not None and rk == 'greedy':
            r['mean_ablation'] = summ(Lm[gi], Lm[0], L[0], t); gi += 1
        res[f'{t}|{rk}|k{k}'] = r
        log(f"{key} {t} {rk} k={k}: binary {100 * r['bin_acc'][0]:.2f} [{100 * r['bin_acc'][1]:.2f}, {100 * r['bin_acc'][2]:.2f}] F {r['F'][0]:.3f} "
            f"[{r['F'][1]:.3f}, {r['F'][2]:.3f}] F_d {r['F_logitdiff'][0]:.3f} 10-class {100 * r['acc10'][0]:.2f} blocks 0-4 {r['frac_blocks_0_4']:.2f}"
            + (f" | mean-abl F {r['mean_ablation']['F'][0]:.3f}" if 'mean_ablation' in r else ''))
    for t in TASKS:
        log(f"{key} {t}: unedited binary {100 * res['unedited'][t]['bin_acc'][0]:.2f}, floor (resample) {100 * res['floor_resample'][f'bin_{t}'][0]:.2f}"
            + (f", floor (mean) {100 * res['floor_mean'][f'bin_{t}'][0]:.2f}" if 'floor_mean' in res else ''))
    out[key] = res; arr[f'{key}_pred'] = L.argmax(-1).astype(np.int8)


def sec_null(lg, S, out, arr):
    ye = S['ye']; sets = circuit_sets(out); rng = np.random.default_rng(5557); res = {}; pb, pe = [], []
    for t in TASKS:
        for k in KS:
            g, e = sets[(t, 'greedy', k)], sets[(t, 'eapig', k)]
            rbm = bm_random(g, N_BM, rng); rbe = bm_random(e, N_BM_EAP, rng); run = [rng.choice(NHEAD, k, replace=False) for _ in range(N_UN)]
            cfgs = [np.ones((NHEAD, 1)), np.zeros((NHEAD, 1)), keep_cfg(g), keep_cfg(e)] + [keep_cfg(h) for h in rbm + rbe + run]
            L = eval_configs(lg, S[f'Be_{t}'], np.stack(cfgs), len(ye)); C = bin_correct(L, ye, t); acc = C.mean(-1)
            F = (acc - acc[1]) / (acc[0] - acc[1])
            i0 = 4; ibm = slice(i0, i0 + N_BM); ibe = slice(i0 + N_BM, i0 + N_BM + N_BM_EAP); iun = slice(i0 + N_BM + N_BM_EAP, None)
            rk = lambda c, sl: float((1 + (acc[sl] >= acc[c]).sum()) / (1 + len(acc[sl])))
            q = lambda v: [float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), float(np.max(v))]
            r = dict(acc_unedited=float(acc[0]), acc_floor=float(acc[1]), acc_greedy=float(acc[2]), acc_eapig=float(acc[3]),
                     F_greedy=faith(C[2], C[1], C[0]), F_eapig=faith(C[3], C[1], C[0]),
                     F_random_blockmatched_greedy=q(F[ibm]), F_random_blockmatched_eapig=q(F[ibe]), F_random_unrestricted=q(F[iun]),
                     p_greedy_vs_blockmatched=rk(2, ibm), p_eapig_vs_blockmatched=rk(3, ibe), p_greedy_vs_unrestricted=rk(2, iun),
                     p_eapig_vs_unrestricted=rk(3, iun), n_random=[N_BM, N_BM_EAP, N_UN])
            res[f'{t}|k{k}'] = r; pb.append(r['p_greedy_vs_blockmatched']); pe.append(r['p_eapig_vs_blockmatched'])
            arr[f'null_{t}_k{k}_F'] = F.astype(np.float32)
            log(f"null {t} k={k}: F greedy {F[2]:.3f} (block-matched random mean {np.mean(F[ibm]):.3f}, 95 % {np.percentile(F[ibm], 97.5):.3f}, "
                f"max {np.max(F[ibm]):.3f}; p {r['p_greedy_vs_blockmatched']:.3f}); F EAP-IG {F[3]:.3f} (bm p {r['p_eapig_vs_blockmatched']:.3f}); "
                f"unrestricted mean {np.mean(F[iun]):.3f} max {np.max(F[iun]):.3f}")
    for nm, pv, fam in (('greedy_vs_blockmatched', pb, 'p_greedy_vs_blockmatched'), ('eapig_vs_blockmatched', pe, 'p_eapig_vs_blockmatched')):
        h = holm(pv)
        for i, kk in enumerate([f'{t}|k{k}' for t in TASKS for k in KS]): res[kk][f'{fam}_holm'] = float(h[i])
    out['null'] = res


def sec_roles(lg, S, out, arr):
    import torch
    ye = S['ye']; sets = circuit_sets(out); res = {}
    for t in TASKS:
        Bt = S[f'Be_{t}']; circ = sets[(t, 'greedy', 16)]
        cf = [np.ones((NHEAD, 1)), np.zeros((NHEAD, 1)), keep_cfg(circ)]
        for h in range(NHEAD): w = np.ones((NHEAD, 1)); w[h] = 0; cf.append(w)
        for h in range(NHEAD): cf.append(keep_cfg([h]))
        for h in circ: cf.append(keep_cfg([x for x in circ if x != h]))
        L = eval_configs(lg, Bt, np.stack(cf), len(ye)); C = bin_correct(L, ye, t); acc = C.mean(-1)
        cu, cfl, cc = C[0], C[1], C[2]
        rem = [dict(head=hn(h), acc=float(acc[3 + h]), dacc=boot_ci(C[3 + h].astype(float) - cu), mcnemar=mcnemar(cu, C[3 + h])) for h in range(NHEAD)]
        kep = [dict(head=hn(h), acc=float(acc[3 + NHEAD + h]), dacc_vs_floor=boot_ci(C[3 + NHEAD + h].astype(float) - cfl), mcnemar=mcnemar(cfl, C[3 + NHEAD + h])) for h in range(NHEAD)]
        for lst in (rem, kep):
            for r, p in zip(lst, holm([r['mcnemar']['p'] for r in lst])): r['p_holm'] = float(p)
        # grade-split within the circuit: (96, 72) weights
        dv = 72; gcols = {g: [c * 16 + j for c in range(4) for j in js] for g, js in GRADES.items()}; gcols['s'] = list(range(64, 72))
        gc = []
        for h in circ:
            for g in ('g0', 'g1', 'g234', 's'):
                w = np.repeat(keep_cfg(circ), dv, 1); w[h, gcols[g]] = 0; gc.append(w)
        Lg = eval_configs(lg, Bt, np.stack(gc), len(ye)); Cg = bin_correct(Lg, ye, t)
        # output energy share per grade (clean pass on task jets)
        ss = torch.zeros(NHEAD, 4, 2, device=lg.dev)
        with torch.no_grad():
            for bt in Bt:
                cap = {}; lg.forward(bt['et'], capture=cap); V = bt['et']['V'].float()
                for b in range(NBLK):
                    o2 = cap[b] ** 2
                    for gi, g in enumerate(('g0', 'g1', 'g234', 's')):
                        x = o2[..., gcols[g]].sum(-1)                                       # (H, B, n)
                        ss[b * NH:(b + 1) * NH, gi, 0] += (x * V[None]).sum((1, 2)); ss[b * NH:(b + 1) * NH, gi, 1] += x[:, :, 0].sum(1)
        sh = (ss / ss.sum(1, keepdim=True)).cpu().numpy()
        heads = []
        for j, h in enumerate(circ):
            d = dict(head=hn(h), removed_alone=rem[h], kept_alone=kep[h], circuit_minus=dict(acc=float(acc[3 + 2 * NHEAD + j]),
                     dacc=boot_ci(C[3 + 2 * NHEAD + j].astype(float) - cc), mcnemar=mcnemar(cc, C[3 + 2 * NHEAD + j])), grade_ablated_in_circuit={},
                     output_share_all_tokens=dict(zip(('g0', 'g1', 'g234', 's'), sh[h, :, 0].tolist())),
                     output_share_global_token=dict(zip(('g0', 'g1', 'g234', 's'), sh[h, :, 1].tolist())))
            for gi, g in enumerate(('g0', 'g1', 'g234', 's')):
                cg = Cg[4 * j + gi]; d['grade_ablated_in_circuit'][g] = dict(acc=float(cg.mean()), dacc=boot_ci(cg.astype(float) - cc), mcnemar=mcnemar(cc, cg))
            heads.append(d)
        pm = holm([d['circuit_minus']['mcnemar']['p'] for d in heads])
        for d, p in zip(heads, pm): d['circuit_minus']['p_holm'] = float(p)
        pg = holm([d['grade_ablated_in_circuit'][g]['mcnemar']['p'] for d in heads for g in ('g0', 'g1', 'g234', 's')])
        for i, (d, g) in enumerate([(d, g) for d in heads for g in ('g0', 'g1', 'g234', 's')]): d['grade_ablated_in_circuit'][g]['p_holm'] = float(pg[i])
        res[t] = dict(acc_unedited=float(acc[0]), acc_floor=float(acc[1]), acc_circuit16=float(acc[2]), removed_alone=rem, kept_alone=kep, circuit16_heads=heads)
        log(f"roles {t}: removed-alone Holm-sig {[r['head'] for r in rem if r['p_holm'] < 0.05]}; kept-alone Holm-sig "
            f"{[r['head'] for r in kep if r['p_holm'] < 0.05]}")
        for d in heads:
            log(f"  {d['head']}: removed alone {100 * d['removed_alone']['dacc'][0]:+.2f}, kept alone {100 * d['kept_alone']['acc']:.2f}, circuit minus "
                f"{100 * d['circuit_minus']['dacc'][0]:+.2f}; grade-ablated " + ' '.join(f"{g} {100 * d['grade_ablated_in_circuit'][g]['dacc'][0]:+.2f}" for g in ('g0', 'g1', 'g234', 's'))
                + '; share ' + ' '.join(f"{g} {v:.2f}" for g, v in d['output_share_all_tokens'].items()))
    out['roles'] = res


def sec_e4(lg, S, out, arr):
    import torch
    from interp.part_core import kin, build, jmass, sub
    t = 'WZ'; ye = S['ye']; Je = S['Je']; sets = circuit_sets(out); rng = np.random.default_rng(5558)
    pt, de, dp, ms = kin(Je); Jd = build(Je, pt, (de * DIL).astype(np.float32), (dp * DIL).astype(np.float32), ms)
    m = task_mask(ye, t); dlm = np.log(np.maximum(jmass(Jd), 1e-3) / np.maximum(jmass(Je), 1e-3))[m]
    Bt = S[f'Be_{t}']
    with torch.no_grad():
        for bt in Bt: bt['ex'] = lg.embed(sub(Jd, bt['idx']))
    names = ['full', 'floor']; cfgs = [np.ones((NHEAD, 1)), np.zeros((NHEAD, 1))]
    for k in KS:
        names.append(f'greedy_k{k}'); cfgs.append(keep_cfg(sets[(t, 'greedy', k)]))
        for i, h in enumerate(bm_random(sets[(t, 'greedy', k)], N_E4, rng)): names.append(f'rand_k{k}_{i}'); cfgs.append(keep_cfg(h))
    cfgs = np.stack(cfgs); L0 = eval_configs(lg, Bt, cfgs, len(ye)); L1 = eval_configs(lg, Bt, cfgs, len(ye), e_key='ex')
    a, b = TASKS[t]; D = (L1[..., a] - L1[..., b] - L0[..., a] + L0[..., b])[:, m]          # (C, n_task) per-jet change of d
    rngb = np.random.default_rng(9); bi = rngb.integers(0, D.shape[1], (NB, D.shape[1]))
    def Rof(j):
        v = D[j].sum() / D[0].sum(); bs = D[j][bi].sum(1) / D[0][bi].sum(1)
        return [float(v), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    res = dict(edit=f'(deta, dphi) x {DIL} about the stored axis for every particle', dlog_m=[float(dlm.mean()), float(np.median(dlm))],
               mean_dd_full=boot_ci(D[0]), mean_dd_full_W=float(D[0][ye[m] == a].mean()), mean_dd_full_Z=float(D[0][ye[m] == b].mean()),
               R_floor=Rof(1), WtoZ_full=[float((L0[0][ye == a].argmax(1) == b).mean()), float((L1[0][ye == a].argmax(1) == b).mean())])
    for k in KS:
        j = names.index(f'greedy_k{k}'); rr = [D[i].sum() / D[0].sum() for i, nm in enumerate(names) if nm.startswith(f'rand_k{k}_')]
        res[f'k{k}'] = dict(R_circuit=Rof(j), R_random_blockmatched=[float(np.mean(rr)), float(np.percentile(rr, 2.5)), float(np.percentile(rr, 97.5))],
                            p_rank_random_ge_circuit=float((1 + sum(x >= D[j].sum() / D[0].sum() for x in rr)) / (1 + len(rr))))
        log(f"e4 k={k}: R circuit {res[f'k{k}']['R_circuit'][0]:.3f} [{res[f'k{k}']['R_circuit'][1]:.3f}, {res[f'k{k}']['R_circuit'][2]:.3f}] "
            f"random block-matched mean {np.mean(rr):.3f}")
    log(f"e4: mean dlog m {dlm.mean():.4f}; mean dd full {res['mean_dd_full'][0]:.3f}; R floor {res['R_floor'][0]:.3f}; W->Z {res['WtoZ_full']}")
    out['e4'] = res
    for bt in Bt: bt.pop('ex', None)


def sec_untrained(S, out, arr, dev):
    import torch
    lu = LGA(None, dev); log('loaded untrained L-GATr')
    Su = dict(S)
    for k in [k for k in S if k.startswith('B')]:
        Su[k] = make_batches(lu, S['Je'] if k.startswith('Be') else (S['Jr'] if k.startswith(('Br', 'Bg')) else S['Jf']),
                             S['idx_' + k], S['don_' + k], S['pool_' + k])
    o = {}; L0 = plain_logits(lu, Su['Be'], len(S['ye']))
    o['base'] = dict(acc10=boot_ci(L0.argmax(1) == S['ye']), **{f'bin_{t}': boot_ci(bin_correct(L0, S['ye'], t)) for t in TASKS})
    sec_rank(lu, Su, o, arr, key='rank', greedy_on=False, write=False)
    sec_circuits(lu, Su, o, arr, L0, key='circuits', mean_variant=False)
    out['untrained'] = o
    del lu; torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------------ figure
def figure(out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'text.usetex': False, 'font.family': 'DejaVu Sans'})
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    lab = {'WZ': 'W vs Z', 'tQCD': 't→bqq′ vs QCD'}
    for i, t in enumerate(TASKS):
        a = ax[i]; N = out['null']; C = out['circuits']
        Fg = np.array([C[f'{t}|greedy|k{k}']['F'] for k in KS]); Fe = np.array([C[f'{t}|eapig|k{k}']['F'] for k in KS])
        bm = np.array([N[f'{t}|k{k}']['F_random_blockmatched_greedy'] for k in KS]); un = np.array([N[f'{t}|k{k}']['F_random_unrestricted'] for k in KS])
        a.fill_between(KS, bm[:, 1], bm[:, 2], color='C7', alpha=0.3, label='random head sets, same heads per block (95 % range)')
        a.plot(KS, bm[:, 0], color='C7', ls='--')
        a.plot(KS, un[:, 0], color='C2', ls=':', marker='.', label='random head sets, any block (mean)')
        a.errorbar(KS, Fg[:, 0], yerr=[Fg[:, 0] - Fg[:, 1], Fg[:, 2] - Fg[:, 0]], color='C0', marker='o', capsize=3, label='top-k heads, greedy ranking')
        a.errorbar(KS, Fe[:, 0], yerr=[Fe[:, 0] - Fe[:, 1], Fe[:, 2] - Fe[:, 0]], color='C1', marker='s', capsize=3, label='top-k heads, EAP-IG ranking')
        a.axhline(1, color='k', lw=0.7); a.axhline(0.8, color='k', lw=0.6, ls=':'); a.axhline(0, color='k', lw=0.5)
        u = 100 * C['unedited'][t]['bin_acc'][0]; f = 100 * C['floor_resample'][f'bin_{t}'][0]
        a.set_title(f'({"ab"[i]}) {lab[t]}: only k heads kept, other {NHEAD}-k resampled\nF = 1: full accuracy {u:.1f} %; F = 0: all heads ablated {f:.1f} %', fontsize=9)
        a.set_xlabel('number of heads kept, k (of 96)'); a.set_ylabel('faithfulness F of binary accuracy'); a.set_xticks(KS); a.legend(fontsize=7)
    x = np.arange(NBLK); w = 0.38
    for j, t in enumerate(TASKS):
        ax[2].bar(x + (j - 0.5) * w, out['circuits'][f'{t}|greedy|k16']['per_block'], w, label=f'{lab[t]} (greedy top 16)')
    ax[2].set_xlabel('L-GATr block'); ax[2].set_ylabel('heads in the k = 16 circuit'); ax[2].set_xticks(x); ax[2].axvline(4.5, color='k', ls=':', lw=0.7)
    ax[2].set_title('(c) where the 16 kept heads sit (P49: decisive heads in blocks 0-4, left of dotted line)', fontsize=9); ax[2].legend(fontsize=7)
    fig.tight_layout(); p = RES / f'p55_lgatr_head_circuit{TAG}.png'; fig.savefig(p, dpi=130); log('wrote', p)


# ------------------------------------------------------------------------------------------------ main
def main():
    out_path = RES / f'p55_lgatr_head_circuit{TAG}.json'
    if ARGS.figure_only:
        figure(json.loads(out_path.read_text())); return
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; secs = ARGS.sections.split(','); log('checkpoint', ARGS.ckpt, 'device', dev)
    Jr, rr, greedy, Jf, fr, Je, er = load_samples(); yr, yf, ye = (J['label'].astype(int) for J in (Jr, Jf, Je))
    assert not set(map(tuple, rr)) & set(map(tuple, fr)), 'ranking and reference rows overlap'
    log(f'samples: ranking {len(yr)} (greedy subset {greedy.sum()}), reference {len(yf)}, evaluation {len(ye)}')
    don_r = pick_donors(Jr['jet_pt'], Jf['jet_pt'], False, 5560); don_e = pick_donors(Je['jet_pt'], Je['jet_pt'], True, 5561)
    dpt = lambda d, J, P: float(np.median(np.abs(np.log(J['jet_pt'] / P['jet_pt'][d]))))
    out = dict(node='P55', ckpt=ARGS.ckpt, N_eval=int(len(ye)),
               samples=dict(ranking=f'test_20M <class>_{RANK_FILE}.root, default_rng({SEED_R}), {PER_RANK} random rows per class for {TCLS}; '
                                    f'reference {PER_REF} other random rows per class (10 classes); greedy subset {PER_GREEDY} per class',
                            evaluation=f'test_20M <class>_{EVAL_FILE}.root, default_rng({SEED_E}), {PER_EVAL} random rows per class, shuffled'),
               donors=dict(rule=f'random among the {K_DONOR} jets nearest in jet pT (any class); ranking: from the reference set; evaluation: same sample, not self',
                           median_abs_dlog_pt_ranking=dpt(don_r, Jr, Jf), median_abs_dlog_pt_eval=dpt(don_e, Je, Je),
                           donor_same_class_frac_eval=float((ye[don_e] == ye).mean())),
               heads='index = block * 8 + head; b<block>h<head>, 0-based', ks=KS)
    arr = dict(rows_rank=rr, rows_ref=fr, rows_eval=er, y_eval=ye, donors_eval=don_e); RES.mkdir(parents=True, exist_ok=True)
    lg = LGA(ARGS.ckpt, dev); log('loaded L-GATr')
    S = dict(Jr=Jr, yr=yr, Jf=Jf, yf=yf, Je=Je, ye=ye, greedy=greedy)
    def mk(name, J, idx, don, pool):
        S['idx_' + name], S['don_' + name], S['pool_' + name] = idx, don, pool; S[name] = make_batches(lg, J, idx, don, pool)
    mk('Be', Je, np.arange(len(ye)), don_e, Je); mk('Bf', Jf, np.arange(len(yf)), np.arange(len(yf)), Jf)
    for t in TASKS:
        mk(f'Br_{t}', Jr, np.where(task_mask(yr, t))[0], don_r, Jf); mk(f'Bg_{t}', Jr, np.where(task_mask(yr, t) & greedy)[0], don_r, Jf)
        mk(f'Be_{t}', Je, np.where(task_mask(ye, t))[0], don_e, Je)
    log('batches built')
    def step(nm, fn):
        try: fn()
        except Exception:
            log(f'SECTION {nm} FAILED'); traceback.print_exc(); out.setdefault('failed', []).append(nm)
        out['wall_s'] = time.time() - T0; jdump(out, out_path); np.savez_compressed(RES / f'p55_arrays{TAG}.npz', **arr)
    L0 = {}
    step('base', lambda: L0.__setitem__('t', sec_base(lg, S, out, arr)))
    if 'means' in secs: step('means', lambda: sec_means(lg, S, out))
    if 'rank' in secs: step('rank', lambda: sec_rank(lg, S, out, arr))
    if 'rank' in out:
        if 'circuits' in secs: step('circuits', lambda: sec_circuits(lg, S, out, arr, L0['t'], mean_variant=lg.means is not None))
        if 'null' in secs: step('null', lambda: sec_null(lg, S, out, arr))
        if 'roles' in secs: step('roles', lambda: sec_roles(lg, S, out, arr))
        if 'e4' in secs: step('e4', lambda: sec_e4(lg, S, out, arr))
    if 'untrained' in secs:
        del lg; torch.cuda.empty_cache(); step('untrained', lambda: sec_untrained(S, out, arr, dev))
    if 'circuits' in out and 'null' in out:
        try: figure(out)
        except Exception: traceback.print_exc()
    log('done')


if __name__ == '__main__':
    main()
