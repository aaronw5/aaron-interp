"""P3 — ParT_full W/Z mass circuit at full statistics with the corrected build (claim C2; Paper C thesis v1 steps 2-3).

Thesis v1 steps 2-3: class-attention head b2h8 switches its class-token share near 85 GeV; with "far-side" heads it writes the
jet mass into the class-token coordinate n98 (W +, Z -). The G-series (G21b/c, G22) measured this on 400 jets/class with the
eta-mirror build. P3 re-measures it and states a literal circuit.

Label mapping (G-series -> here): class-attention head bXhY = model.cls_blocks[X-1], head Y-1 (head h = dims 16h..16h+15 of
the MultiheadAttention in-projection); b2h8 = (1, 7). Neuron nK = index K-1 of z, the class token after the final LayerNorm
(ParTNetwork.run's z; no activation follows; logits = z W + b); n98 = index 97 (part_core.N98). Particle-block heads "far side"
(G101) = heads 4 and 8 of each of the 8 particle blocks = indices (i, 3), (i, 7).

Samples (JetClass validation split, /j-jepa-vol/JetClass/Pythia/val_5M; random rows, numpy default_rng(31), never first-k):
  S: 2,000 random rows of each class's <sample>_120.root (20,000 jets) -> head roles (b), circuits (c), all-class span.
  T: 10,000 random rows each of WToQQ_121.root and ZToQQ_121.root (20,000 W/Z jets) -> threshold (a), edits, G21b decomposition;
     T[::5] (4,000 jets) -> particle-block far-side patch.
  P1 sample (test files, rows 3,000-4,000 per class) -> unedited accuracy check only.
Build: part_core.build (eta = jet_eta + sign(jet_eta) * deta). Jet mass = jmass of the built jet [GeV].

Model internals: the particle blocks run as weaver runs them (input of cls_blocks[0] captured by a pre-hook); the two
class-attention blocks and the head are re-implemented here exactly as weaver-core 0.4.17's Block (query = the raw class token,
keys/values = pre_attn_norm(cat(cls, particles)), out_proj, the c_attn per-'head' scale applied after out_proj with weaver's
'tbhd,h->tbdh' reshuffle, post_attn_norm, residual, pre_fc_norm, fc1, GELU, post_fc_norm, fc2, w_resid * residual), checked
against the model's own logits on every batch. Interventions in the re-implementation:
  head context ctx_bh = sum_p a_bhp v_bhp (16-dim, before out_proj): replaced (resample / mean) or shifted (random direction);
  attention weights a_bh (key 0 = the class token: 'class-token share'): replaced (G21b decomposition);
  MLP unit = post_fc_norm output j of a class block (the coefficient of fc2 column j; 512 per block): replaced;
  z coordinate: replaced.
Particle-block head output o_ih = (A_ih V_ih) W_o[:, 16h:16h+16]^T (exact: sum_h o_ih + b_o = the attention output, checked):
patched by wrapping MultiheadAttention.forward.

(a) Threshold: per class-attention head, class-token share vs jet mass on T (5 GeV bins, 1,000 bootstrap resamples); logistic
    s(m) = lo + (hi - lo) / (1 + exp((m - m50)/w)) fitted to b2h8 per jet on 50-140 GeV (300 bootstrap fits); span (max - min of
    the binned mean over 50-130 GeV bins with >= 50 jets) per head; H0: span <= 20 pt, H1: span > 20 pt, bootstrap p, Holm over
    16 heads. All-class span on S (40-240 GeV) as G22. Causal: edits angles x 0.9 / x 1.1 (every particle's deta, dphi x k;
    true mass x ~k) and phi + 0.06 rad (P1; invariant mass unchanged): delta share per head on T, bootstrap CI, Holm; the
    observational curve's prediction for the edit (curve(m_edit) - curve(m_base)) as the calibration.
(b) Head -> n98 on S: per class-attention head, ctx (i) resampled from a same-class jet whose mass differs by >= 10 GeV (one
    donor per jet, seed 31), (ii) set to its mean over S, (iii) shifted by a random 16-dim direction of the same norm as the
    resample change (NDIR draws). Metrics on the 4,000 W/Z jets of S: slope of delta z97 [s.d.] and of delta(lW - lZ) on the
    donor - own mass difference [per 10 GeV] (OLS through 0), mean |delta z97|; share of the all-16-heads resample slope;
    mean ablation: mean delta z97 and delta(lW - lZ) on W and Z, W/Z binary calls changed, McNemar. Bootstrap over jets, Holm
    over heads. G21b decomposition on T (W/Z, base mass 70-110 GeV) for angles x 1.1 and phi + 0.06: delta z97 = attention
    change (all heads' a from the edited run, values from the base run) + value change (base a, edited values) + interaction;
    per head: only that head's a from the edited run. Particle-block far-side heads (h4, h8 of all 8 blocks) patched from the
    unedited jet under the edit (G103 style) vs NPSET random sets of 16 of the 64 particle heads, on T[::5].
(c) Literal circuits on S, ParT_full (resample distribution: one donor per jet drawn uniformly from S excluding itself, seed 31,
    the same donor for every ablated component and every circuit; ablated components take the donor's clean values; particle
    blocks intact = the circuit's scope is the class-attention stage and the read-out):
    V_a heads {b2h8} + F (F = heads named by (b)), every class-block MLP unit resampled, z: only n98 kept;
    V_b as V_a + the 16 MLP units per block with the largest |grad z97 * (h - mean h)| on W/Z jets kept;
    V_c heads {b2h8} + F, all MLP units kept, z: only n98 kept;
    V_d heads {b2h8} + F, everything else in the class stage and read-out kept;
    V_e n98 alone (whole class stage kept, z: only n98) vs each of the other 127 single coordinates.
    Each: 10-class accuracy and confusion, W-vs-Z binary accuracy (true W/Z jets, argmax over (lW, lZ)), same call as unedited;
    vs NRAND random circuits of equal size (random heads from the 16, random MLP units, random other active z coordinate);
    permutation p = (1 + #random >= circuit) / (1 + NRAND).
ParT_kin as second model: (a) and (b) (Delta(lW - lZ) metrics; index 97 is not a W/Z coordinate in ParT_kin).

  ./part_job.sh c-p3-run 'python -m interp.p3_wz_circuit_part'          -> results/part/p3_wz_circuit.json (+ figure)
  ./part_job.sh c-p3-smoke 'python -m interp.p3_wz_circuit_part --smoke' -> results/part/p3_wz_circuit_smoke.json
  python -m interp.p3_wz_circuit_part --plot                              (figure from the json; matplotlib only)
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np

OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
VAL = Path(os.environ.get('P3_VAL', '/j-jepa-vol/JetClass/Pythia/val_5M'))
SMOKE = '--smoke' in sys.argv
SEED = 31
PER, PER_T = (100, 400) if SMOKE else (2000, 10000)
NB = 200 if SMOKE else 1000
NFIT = 30 if SMOKE else 300
NDIR = 3 if SMOKE else 20
NRAND = 8 if SMOKE else 100
NPSET = 3 if SMOKE else 20
H, HD, C = 8, 16, 128
B2H8 = (1, 7)
HEADS = [(b, h) for b in range(2) for h in range(H)]
FAR_P = [(i, h) for i in range(8) for h in (3, 7)]
T0 = time.time()
lab = lambda bh: f'b{bh[0] + 1}h{bh[1] + 1}'
FN = OUT / ('p3_wz_circuit_smoke.json' if SMOKE else 'p3_wz_circuit.json')


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


def p_two(bs, v0=0.0):
    bs = np.asarray(bs, float); nb = len(bs)
    return float(max(1.0 / nb, min(1.0, 2 * min((bs <= v0).mean(), (bs >= v0).mean()))))


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.zeros(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj.tolist()


def mcnemar(ok0, ok1):
    """exact two-sided McNemar on paired correctness: b = right->wrong, c = wrong->right"""
    from scipy.stats import binomtest
    b = int((ok0 & ~ok1).sum()); c = int((~ok0 & ok1).sum())
    return dict(right_to_wrong=b, wrong_to_right=c, p=(float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0))


def slope_stats(Wb, dy, dx):
    """OLS slope through 0 of dy on dx, with bootstrap (rows of Wb = resample counts)"""
    s = float((dy * dx).sum() / (dx * dx).sum()); bs = (Wb @ (dy * dx)) / (Wb @ (dx * dx))
    return s, bs


def mean_stats(Wb, x):
    return float(x.mean()), (Wb @ x) / Wb.sum(1)


# ------------------------------------------------------------------------------------------------------------ data
def load_rows(path, n, rng, chunk=25000):
    import uproot
    from interp.part_core import read_root, sub, cat
    N = uproot.open(path)['tree'].num_entries; rows = np.sort(rng.choice(N, n, replace=False)); parts = []
    for s in range(0, N, chunk):
        r = rows[(rows >= s) & (rows < s + chunk)]
        if len(r): parts.append(sub(read_root(path, s, min(s + chunk, N)), r - s))
    return cat(parts), rows, N


# ------------------------------------------------------------------------------------------------------------ model stage
class Stage:
    def __init__(self, name):
        import torch, torch.nn.functional as F
        from interp.part_core import ParTNetwork
        self.t, self.F, self.name = torch, F, name
        self.net = ParTNetwork(name); self.m = self.net.model; self.dev = self.net.device
        g = lambda p: p.detach().float()
        ln = lambda l: (g(l.weight), g(l.bias), l.eps)
        self.cls0 = g(self.m.cls_token).reshape(C); self.B = []
        for blk in self.m.cls_blocks:
            a = blk.attn; Wi, bi = g(a.in_proj_weight), g(a.in_proj_bias)
            self.B.append(dict(ln_pre=ln(blk.pre_attn_norm), Wq=Wi[:C], bq=bi[:C], Wk=Wi[C:2 * C], bk=bi[C:2 * C], Wv=Wi[2 * C:], bv=bi[2 * C:],
                               Wo=g(a.out_proj.weight), bo=g(a.out_proj.bias), c=g(blk.c_attn), ln_pa=ln(blk.post_attn_norm), ln_pf=ln(blk.pre_fc_norm),
                               W1=g(blk.fc1.weight), b1=g(blk.fc1.bias), ln_po=ln(blk.post_fc_norm), W2=g(blk.fc2.weight), b2=g(blk.fc2.bias), wr=g(blk.w_resid)))
        self.lnf = ln(self.m.norm); self.Wfc, self.bfc = g(self.m.fc[0].weight), g(self.m.fc[0].bias)
        self.cap = {}; self.m.cls_blocks[0].register_forward_pre_hook(lambda mod, args: self.cap.__setitem__('X', args[0]))
        self.pm, self.rec, self.recon = None, {}, 0.0
        for i, blk in enumerate(self.m.blocks): self._wrap(i, blk.attn)

    def _wrap(self, i, attn):
        orig, S, t = attn.forward, self, self.t

        def f(query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True):
            if S.pm is None:
                return orig(query, key, value, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask,
                            average_attn_weights=average_attn_weights)
            out, w = orig(query, key, value, key_padding_mask=key_padding_mask, need_weights=True, attn_mask=attn_mask, average_attn_weights=False)
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
        """particle blocks on jets J -> X (n, P, C), pad (n, P) bool (True = padded), model logits, model z"""
        from interp.part_core import inputs
        t = self.t; x, v, m = inputs(J, self.name)
        with t.no_grad():
            L = self.m(*(t.from_numpy(a).to(self.dev) for a in (x, v, m)))
        X = self.cap['X'].permute(1, 0, 2).contiguous().float(); pad = t.from_numpy(m[:, 0] == 0).to(self.dev)
        assert X.shape[1] == pad.shape[1]
        return X, pad, L.float(), self.net._cls['h'].float()

    def cls(self, X, pad, ctl=None, rec=False):
        """the two class-attention blocks + final norm + fc on particle embeddings X; ctl: interventions (see docstring)"""
        t, F = self.t, self.F; ctl = ctl or {}; n, P, _ = X.shape
        r = self.cls0.expand(n, C)
        padf = t.cat([t.zeros(n, 1, dtype=t.bool, device=X.device), pad], 1)
        R = dict(A=[], ctx=[], h=[])
        for b, B in enumerate(self.B):
            u = t.cat([r[:, None], X], 1); u = F.layer_norm(u, (C,), B['ln_pre'][0], B['ln_pre'][1], B['ln_pre'][2])
            K = (u @ B['Wk'].T + B['bk']).view(n, P + 1, H, HD); V = (u @ B['Wv'].T + B['bv']).view(n, P + 1, H, HD)
            q = (r @ B['Wq'].T + B['bq']).view(n, H, HD) / math.sqrt(HD)
            A = t.softmax(t.einsum('nhd,nphd->nhp', q, K).masked_fill(padf[:, None, :], float('-inf')), -1)
            if ('attn', b) in ctl:
                Ao, sel = ctl[('attn', b)]; A = t.where(sel[None, :, None], Ao, A)
            ctx = t.einsum('nhp,nphd->nhd', A, V)
            if ('ctx', b) in ctl:
                Co, sel = ctl[('ctx', b)]; ctx = t.where(sel[None, :, None], Co, ctx)
            if ('ctx_add', b) in ctl: ctx = ctx + ctl[('ctx_add', b)]
            o = ctx.reshape(n, C) @ B['Wo'].T + B['bo']
            o = (o.view(n, H, HD) * B['c'][None, :, None]).permute(0, 2, 1).reshape(n, C)
            x1 = F.layer_norm(o, (C,), *B['ln_pa'][:2], B['ln_pa'][2]) + r
            hh = F.gelu(F.layer_norm(x1, (C,), *B['ln_pf'][:2], B['ln_pf'][2]) @ B['W1'].T + B['b1'])
            hh = F.layer_norm(hh, (hh.shape[-1],), *B['ln_po'][:2], B['ln_po'][2])
            if ('mlp', b) in ctl:
                Ho, sel = ctl[('mlp', b)]; hh = t.where(sel[None, :], Ho, hh)
            if ('mlp_add', b) in ctl: hh = hh + ctl[('mlp_add', b)]
            r = hh @ B['W2'].T + B['b2'] + B['wr'] * x1
            if rec: R['A'].append(A); R['ctx'].append(ctx); R['h'].append(hh)
        z = F.layer_norm(r, (C,), *self.lnf[:2], self.lnf[2])
        if 'z' in ctl:
            Zo, sel = ctl['z']; z = t.where(sel[None, :], Zo, z)
        return z, z @ self.Wfc.T + self.bfc, R


def batches(J, size):
    order = np.argsort(J['mask'].sum(1), kind='stable')
    return [order[i:i + size] for i in range(0, len(order), size)]


# ------------------------------------------------------------------------------------------------------------ S: base pass
def base_pass(st, J, mass):
    from interp.part_core import sub
    t = st.t; N = len(J['mask']); dR = np.hypot(J['deta'], J['dphi']).astype(np.float32)
    D = dict(ctx=np.zeros((N, 2, H, HD), np.float32), h=np.zeros((N, 2, 512), np.float32), z=np.zeros((N, C), np.float32),
             L=np.zeros((N, 10), np.float32), share=np.zeros((N, 2, H), np.float32), dRw=np.zeros((N, 2, H), np.float32),
             dRu=np.zeros(N, np.float32), bat=[])
    md, calls = 0.0, 0
    for idx in batches(J, 256):
        X, pad, Lm, zm = st.run_X(sub(J, idx))
        with t.no_grad():
            z, L, R = st.cls(X, pad, rec=True)
        md = max(md, float((L - Lm).abs().max())); calls += int((L.argmax(1) != Lm.argmax(1)).sum())
        A = t.stack(R['A'], 1)                                         # (n, 2, H, P+1)
        Ap = A[..., 1:]; P = X.shape[1]; d = t.from_numpy(dR[idx, :P]).to(st.dev)
        D['dRw'][idx] = ((Ap * d[:, None, None, :]).sum(-1) / Ap.sum(-1).clamp_min(1e-12)).cpu().numpy()
        real = (~pad).float(); D['dRu'][idx] = ((d * real).sum(1) / real.sum(1)).cpu().numpy()
        D['share'][idx] = A[..., 0].cpu().numpy(); D['ctx'][idx] = t.stack(R['ctx'], 1).cpu().numpy()
        D['h'][idx] = t.stack(R['h'], 1).cpu().numpy(); D['z'][idx] = z.cpu().numpy(); D['L'][idx] = L.cpu().numpy()
        D['bat'].append(dict(idx=idx, it=t.from_numpy(idx).to(st.dev), X=X, pad=pad))
    D['reimpl_max_dlogit'], D['reimpl_calls_changed'] = md, calls
    D['T'] = {k: t.from_numpy(D[k]).to(st.dev) for k in ('ctx', 'h', 'z')}
    return D


def eval_S(st, D, mk):
    """class stage over every S batch with ctl = mk(idx_tensor) -> z (N, C), L (N, 10)"""
    t = st.t; N = len(D['L']); Z = np.zeros((N, C), np.float32); L = np.zeros((N, 10), np.float32)
    with t.no_grad():
        for bt in D['bat']:
            z, l, _ = st.cls(bt['X'], bt['pad'], mk(bt['it']))
            Z[bt['idx']] = z.cpu().numpy(); L[bt['idx']] = l.cpu().numpy()
    return Z, L


# ------------------------------------------------------------------------------------------------------------ (b) heads on S
def heads_part(st, D, y, mass, rng, iW, iZ, N98, full, ref):
    """ref: dict(z97=, wz=) real-edit slopes per 10 GeV (angles x 1.1 on T) used as the calibration denominator"""
    t = st.t; N = len(y); wz = np.isin(y, [iW, iZ]); nwz = int(wz.sum())
    sd97 = float(D['z'][:, N98].std()); dWZ0 = D['L'][:, iW] - D['L'][:, iZ]
    # donors: same class, |dm| >= 10 GeV
    don = np.zeros(N, int)
    for c in np.unique(y):
        ix = np.where(y == c)[0]
        for i in ix:
            cand = ix[np.abs(mass[ix] - mass[i]) >= 10.0]
            don[i] = rng.choice(cand) if len(cand) else rng.choice(ix[ix != i])
    dT = t.from_numpy(don).to(st.dev); dm = (mass[don] - mass) / 10.0
    Wb = boot_W(nwz, NB, rng)
    ctx_mean = D['T']['ctx'].mean(0)                                    # (2, H, HD)
    selm = lambda hs, b: t.tensor([(b, h) in hs for h in range(H)], device=st.dev)

    def resample(hs):
        return lambda it: {('ctx', b): (D['T']['ctx'][dT[it]][:, b], selm(hs, b)) for b in range(2) if any(bb == b for bb, _ in hs)}

    def meanabl(hs):
        return lambda it: {('ctx', b): (ctx_mean[b][None].expand(len(it), H, HD), selm(hs, b)) for b in range(2) if any(bb == b for bb, _ in hs)}

    def metr(Z, L):
        dz = (Z[:, N98] - D['z'][:, N98]) / sd97; dw = (L[:, iW] - L[:, iZ]) - dWZ0
        return dz, dw

    def summarize(dz, dw, L, kind):
        e = {}
        x = dm[wz]
        for key, v in (('z97', dz[wz]), ('wz', dw[wz])):
            s, bs = slope_stats(Wb, v, x); e[f'slope_{key}'] = s; e[f'slope_{key}_ci'] = ci(bs); e[f'slope_{key}_p0'] = p_two(bs)
            e[f'_bs_{key}'] = bs
            m_, bm = mean_stats(Wb, np.abs(v)); e[f'meanabs_{key}'] = m_; e[f'meanabs_{key}_ci'] = ci(bm)
        for cn, ci_ in (('W', iW), ('Z', iZ)):
            sel = y == ci_
            e[f'mean_dz97_{cn}'] = float(dz[sel].mean()); e[f'mean_dwz_{cn}'] = float(dw[sel].mean())
            e[f'mean_dz97_{cn}_ci'] = ci([dz[sel][rng.integers(0, sel.sum(), sel.sum())].mean() for _ in range(200)])
            e[f'mean_dwz_{cn}_ci'] = ci([dw[sel][rng.integers(0, sel.sum(), sel.sum())].mean() for _ in range(200)])
        c0 = np.where(dWZ0 > 0, iW, iZ); c1 = np.where((L[:, iW] - L[:, iZ]) > 0, iW, iZ)
        e['wz_binary_changed'] = float((c0[wz] != c1[wz]).mean())
        e['wz_binary_acc'] = float((c1[wz] == y[wz]).mean()); e['acc10'] = float((L.argmax(1) == y).mean())
        e['mcnemar_wz'] = mcnemar(c0[wz] == y[wz], c1[wz] == y[wz])
        return e

    res = dict(sd97=sd97, n_wz=nwz, donor_rule='same class, |m_donor - m| >= 10 GeV, seed 31',
               donor_dm_abs_median=float(np.median(np.abs(mass[don] - mass)[wz])), heads={}, groups={})
    groups = {'all16': HEADS, 'block1': [(0, h) for h in range(H)], 'block2': [(1, h) for h in range(H)],
              'b2h8_plus_all_block1': [(0, h) for h in range(H)] + [B2H8]}
    for g, hs in groups.items():
        Z, L = eval_S(st, D, resample(hs)); dz, dw = metr(Z, L); e = summarize(dz, dw, L, 'resample')
        Zm, Lm = eval_S(st, D, meanabl(hs)); dzm, dwm = metr(Zm, Lm); em = summarize(dzm, dwm, Lm, 'mean')
        res['groups'][g] = dict(resample=e, mean=em)
        log(f'  [{st.name}] group {g}: resample slope z97 {e["slope_z97"]:+.4f} {e["slope_z97_ci"]}, slope lW-lZ {e["slope_wz"]:+.4f}; '
            f'mean-abl W/Z calls changed {em["wz_binary_changed"]:.4f}, acc10 {em["acc10"]:.4f}')
    tot_z, tot_w = res['groups']['all16']['resample']['_bs_z97'], res['groups']['all16']['resample']['_bs_wz']
    tz, tw = res['groups']['all16']['resample']['slope_z97'], res['groups']['all16']['resample']['slope_wz']
    for bh in HEADS:
        Z, L = eval_S(st, D, resample([bh])); dz, dw = metr(Z, L); e = summarize(dz, dw, L, 'resample')
        e['share_z97'] = e['slope_z97'] / tz; e['share_z97_ci'] = ci(e['_bs_z97'] / tot_z)
        e['share_wz'] = e['slope_wz'] / tw; e['share_wz_ci'] = ci(e['_bs_wz'] / tot_w)
        e['p_share_z97_ge_0.5'] = float(max(1.0 / NB, ((e['_bs_z97'] / tot_z) >= 0.5).mean()))
        for key in ('z97', 'wz'):
            if ref.get(key):
                e[f'cal_{key}'] = e[f'slope_{key}'] / ref[key]; e[f'cal_{key}_ci'] = ci(e[f'_bs_{key}'] / ref[key])
                e[f'p_cal_{key}_ge_0.5'] = float(max(1.0 / NB, ((e[f'_bs_{key}'] / ref[key]) >= 0.5).mean()))
        Zm, Lm = eval_S(st, D, meanabl([bh])); dzm, dwm = metr(Zm, Lm); em = summarize(dzm, dwm, Lm, 'mean')
        # random 16-dim directions of the resample change's norm
        b, h = bh; dctx = (D['T']['ctx'][dT][:, b, h] - D['T']['ctx'][:, b, h]).norm(dim=-1)   # (N,)
        rnd = []
        for k in range(NDIR):
            g_ = t.Generator(device=st.dev).manual_seed(1000 + 37 * k + 101 * (b * H + h))
            dirs = t.randn(N, HD, generator=g_, device=st.dev); dirs = dirs / dirs.norm(dim=-1, keepdim=True) * dctx[:, None]

            def mk(it, dirs=dirs, b=b, h=h):
                add = t.zeros(len(it), H, HD, device=st.dev); add[:, h] = dirs[it]; return {('ctx_add', b): add}
            Zr, Lr = eval_S(st, D, mk); dzr, dwr = metr(Zr, Lr)
            rnd.append(dict(slope_z97=float((dzr[wz] * dm[wz]).sum() / (dm[wz] ** 2).sum()), meanabs_z97=float(np.abs(dzr[wz]).mean()),
                            slope_wz=float((dwr[wz] * dm[wz]).sum() / (dm[wz] ** 2).sum()), meanabs_wz=float(np.abs(dwr[wz]).mean())))
        R_ = {k: [r[k] for r in rnd] for k in rnd[0]}
        e['random_dir'] = {k: dict(mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v))) for k, v in R_.items()}
        e['p_meanabs_z97_vs_dir'] = (1 + sum(v >= e['meanabs_z97'] for v in R_['meanabs_z97'])) / (1 + NDIR)
        e['p_absslope_z97_vs_dir'] = (1 + sum(abs(v) >= abs(e['slope_z97']) for v in R_['slope_z97'])) / (1 + NDIR)
        e['p_meanabs_wz_vs_dir'] = (1 + sum(v >= e['meanabs_wz'] for v in R_['meanabs_wz'])) / (1 + NDIR)
        res['heads'][lab(bh)] = dict(resample=e, mean=em, dR_attn_weighted_W=float(D['dRw'][y == iW, b, h].mean()),
                                    dR_attn_weighted_Z=float(D['dRw'][y == iZ, b, h].mean()), share_W=float(D['share'][y == iW, b, h].mean()),
                                    share_Z=float(D['share'][y == iZ, b, h].mean()))
        log(f'  [{st.name}] {lab(bh)}: slope z97 {e["slope_z97"]:+.4f} {np.round(e["slope_z97_ci"], 4).tolist()} share {e["share_z97"]:+.3f} '
            f'{np.round(e["share_z97_ci"], 3).tolist()} cal {e.get("cal_z97", float("nan")):+.3f} | slope lW-lZ {e["slope_wz"]:+.4f} share {e["share_wz"]:+.3f} | mean|dz97| {e["meanabs_z97"]:.4f} '
            f'(rand dir {e["random_dir"]["meanabs_z97"]["mean"]:.4f}, max {e["random_dir"]["meanabs_z97"]["max"]:.4f}) | mean-abl dz97 W {em["mean_dz97_W"]:+.3f} '
            f'Z {em["mean_dz97_Z"]:+.3f}, dWZ W {em["mean_dwz_W"]:+.3f} Z {em["mean_dwz_Z"]:+.3f}, W/Z calls changed {em["wz_binary_changed"]:.4f} '
            f'McNemar p {em["mcnemar_wz"]["p"]:.3g}')
    # Holm over heads
    for key in ('slope_z97', 'slope_wz'):
        ps = [res['heads'][lab(bh)]['resample'][f'{key}_p0'] for bh in HEADS]
        for bh, pa in zip(HEADS, holm(ps)): res['heads'][lab(bh)]['resample'][f'{key}_p0_holm'] = pa
    ps = [res['heads'][lab(bh)]['mean']['mcnemar_wz']['p'] for bh in HEADS]
    for bh, pa in zip(HEADS, holm(ps)): res['heads'][lab(bh)]['mean']['mcnemar_wz_p_holm'] = pa
    key = 'z97' if full else 'wz'
    for g in res['groups'].values():
        for k2 in ('z97', 'wz'):
            if ref.get(k2): g['resample'][f'cal_{k2}'] = g['resample'][f'slope_{k2}'] / ref[k2]
    res['ref_edit_slopes_per_10GeV'] = ref
    rank = sorted(HEADS, key=lambda bh: -abs(res['heads'][lab(bh)]['resample'][f'cal_{key}']))
    res['rank_by_abs_share'] = [lab(bh) for bh in rank]; res['rank_metric'] = f'|cal_{key}| = |resample slope / real-edit slope|'
    named = [bh for bh in rank if bh != B2H8 and res['heads'][lab(bh)]['resample'][f'slope_{key}_p0_holm'] < 0.05
             and abs(res['heads'][lab(bh)]['resample'][f'cal_{key}']) >= 0.10][:4]
    res['named_heads_F'] = [lab(bh) for bh in named]
    res['named_rule'] = f'heads other than b2h8 with Holm p < 0.05 for slope_{key} != 0 and |slope_{key}| >= 0.10 x |real-edit slope (angles x 1.1, per 10 GeV)|, by |slope|, at most 4'
    for e in [res['heads'][k]['resample'] for k in res['heads']] + [res['groups'][g][v] for g in res['groups'] for v in ('resample', 'mean')]:
        for k in [k for k in e if k.startswith('_bs')]: del e[k]
    res['_donor'] = don; res['_named'] = named
    log(f'  [{st.name}] rank by |share|: {res["rank_by_abs_share"][:6]}; named F = {res["named_heads_F"]}')
    return res


# ------------------------------------------------------------------------------------------------------------ (c) circuits
def circuits_part(st, D, y, rng, iW, iZ, N98, named):
    from interp.part_core import ACTIVE
    t = st.t; N = len(y); wz = np.isin(y, [iW, iZ])
    don = rng.integers(0, N - 1, N); don = don + (don >= np.arange(N))      # uniform over S without self
    iwz = np.where(wz)[0]; don_wz = iwz[rng.integers(0, len(iwz), N)]
    for i in np.where(don_wz == np.arange(N))[0]: don_wz[i] = iwz[(np.searchsorted(iwz, i) + 1) % len(iwz)]
    DON = dict(uniform_S=don, WZ_only=don_wz); dT = None
    c0 = D['L'].argmax(1); acc0 = float((c0 == y).mean()); b0 = np.where(D['L'][:, iW] > D['L'][:, iZ], iW, iZ)
    # MLP attribution |grad z97 * (h - mean h)| on W/Z jets
    hmean = D['T']['h'].mean(0); attr = t.zeros(2, 512, device=st.dev); wzT = t.from_numpy(wz).to(st.dev)
    for bt in D['bat']:
        it = bt['it']; add = t.zeros(len(it), 2, 512, device=st.dev, requires_grad=True)
        with t.enable_grad():
            z, _, _ = st.cls(bt['X'], bt['pad'], {('mlp_add', 0): add[:, 0], ('mlp_add', 1): add[:, 1]})
            (z[:, N98] * wzT[it].float()).sum().backward()
        attr += ((add.grad * (D['T']['h'][it] - hmean)).abs() * wzT[it].float()[:, None, None]).sum(0)
    attr = attr.cpu().numpy()
    top = np.zeros((2, 512), bool)
    for b in range(2): top[b, np.argsort(-attr[b])[:16]] = True
    Wb10 = boot_W(N, NB, rng); Wbwz = boot_W(int(wz.sum()), NB, rng)

    def run(heads, mlp_keep, z_keep):
        hm = [t.tensor([(b, h) not in heads for h in range(H)], device=st.dev) for b in range(2)]
        mm = [t.from_numpy(~mlp_keep[b]).to(st.dev) for b in range(2)]; zm = t.from_numpy(~z_keep).to(st.dev)

        def mk(it):
            d = DT['d'][it]; ctl = {}
            for b in range(2):
                if bool(hm[b].any()): ctl[('ctx', b)] = (D['T']['ctx'][d][:, b], hm[b])
                if bool(mm[b].any()): ctl[('mlp', b)] = (D['T']['h'][d][:, b], mm[b])
            if bool(zm.any()): ctl['z'] = (D['T']['z'][d], zm)
            return ctl
        _, L = eval_S(st, D, mk); return L

    def metr(L, full_stats=False):
        c = L.argmax(1); bw = np.where(L[:, iW] > L[:, iZ], iW, iZ)
        e = dict(acc10=float((c == y).mean()), wz_binary_acc=float((bw[wz] == y[wz]).mean()), same_call=float((c == c0).mean()),
                 wz_binary_same_as_unedited=float((bw[wz] == b0[wz]).mean()))
        if full_stats:
            ok = (c == y).astype(np.float32); okb = (bw[wz] == y[wz]).astype(np.float32)
            e['acc10_ci'] = ci((Wb10 @ ok) / N); e['wz_binary_acc_ci'] = ci((Wbwz @ okb) / wz.sum())
            e['confusion'] = np.array([[int(((y == i) & (c == j)).sum()) for j in range(10)] for i in range(10)]).tolist()
            e['mcnemar_wz_vs_unedited'] = mcnemar(b0[wz] == y[wz], bw[wz] == y[wz])
        return e

    allm = np.ones((2, 512), bool); nom = np.zeros((2, 512), bool); allz = np.ones(C, bool); z97 = np.zeros(C, bool); z97[N98] = True
    Hc = [B2H8] + list(named)
    V = {'V_a_heads_n98_strict': (Hc, nom, z97), 'V_b_heads_32mlp_n98': (Hc, top, z97), 'V_c_heads_allmlp_n98': (Hc, allm, z97),
         'V_d_heads_only': (Hc, allm, allz), 'V_e_n98_only': (HEADS, allm, z97), 'b2h8_only_heads': ([B2H8], allm, allz),
         'unedited_through_circuit_code': (HEADS, allm, allz)}
    others = [a for a in ACTIVE if a != N98]
    res = dict(unedited=dict(acc10=acc0, wz_binary_acc=float((b0[wz] == y[wz]).mean()),
                             confusion=np.array([[int(((y == i) & (c0 == j)).sum()) for j in range(10)] for i in range(10)]).tolist()),
               circuit_heads=[lab(bh) for bh in Hc], mlp_top16_per_block=[np.where(top[b])[0].tolist() for b in range(2)],
               mlp_attr_top16_share=[float(np.sort(attr[b])[::-1][:16].sum() / attr[b].sum()) for b in range(2)],
               donor_rule=dict(uniform_S='one donor per jet, uniform over the 20,000 S jets without itself, seed 31',
                               WZ_only='one donor per jet, uniform over the 4,000 W/Z jets of S without itself, seed 31',
                               common='same donor for all ablated components and for every circuit (paired)'),
               variants={})
    DT = {}
    for dname, dd in DON.items():
      DT['d'] = t.from_numpy(dd).to(st.dev); res['variants'][dname] = {}
      for name, (hs, mk_, zk) in V.items():
          L = run(hs, mk_, zk); e = metr(L, True); e['size'] = dict(heads=len(hs), mlp_units=int(mk_.sum()), z_coords=int(zk.sum()))
          rnd = []
          if name not in ('unedited_through_circuit_code',):
              for k in range(NRAND):
                  if len(hs) < len(HEADS):
                      while True:
                          sel = rng.choice(len(HEADS), len(hs), replace=False); rh = [HEADS[i] for i in sel]
                          if set(rh) != set(hs): break
                  else: rh = HEADS
                  if mk_.all() or not mk_.any(): rm = mk_
                  else:
                      rm = np.zeros((2, 512), bool)
                      for b in range(2): rm[b, rng.choice(512, int(mk_[b].sum()), replace=False)] = True
                  if zk.all(): rz = zk
                  else: rz = np.zeros(C, bool); rz[rng.choice(others)] = True
                  if rh == hs and (rm == mk_).all() and (rz == zk).all(): continue
                  rnd.append(metr(run(rh, rm, rz)))
          if rnd:
              for k in ('acc10', 'wz_binary_acc'):
                  v = np.array([r[k] for r in rnd]); e[f'random_{k}'] = dict(n=len(v), mean=float(v.mean()), p95=float(np.percentile(v, 95)), max=float(v.max()),
                                                                               min=float(v.min()), values=v.tolist())
                  e[f'p_perm_{k}'] = float((1 + (v >= e[k]).sum()) / (1 + len(v)))
          res['variants'][dname][name] = e
          log(f'  [circuit {dname}] {name}: heads {[lab(b) for b in hs] if len(hs) < 16 else "all"} mlp {int(mk_.sum())} z {int(zk.sum())} | acc10 {e["acc10"]:.4f} '
              f'{np.round(e["acc10_ci"], 4).tolist()} | W/Z binary {e["wz_binary_acc"]:.4f} | same call {e["same_call"]:.4f}'
              + (f' | random W/Z binary mean {e["random_wz_binary_acc"]["mean"]:.4f} max {e["random_wz_binary_acc"]["max"]:.4f} p {e["p_perm_wz_binary_acc"]:.3f}'
                 f' | random acc10 mean {e["random_acc10"]["mean"]:.4f} p {e["p_perm_acc10"]:.3f}' if rnd else ''))
    # single z coordinate read-outs (class stage intact): every coordinate
    Wfc = st.Wfc.cpu().numpy(); bfc = st.bfc.cpu().numpy(); Zd = D['z'][don]; sing = []
    for k in range(C):
        Zm = Zd.copy(); Zm[:, k] = D['z'][:, k]; L = Zm @ Wfc.T + bfc
        sing.append(float((np.where(L[:, iW] > L[:, iZ], iW, iZ)[wz] == y[wz]).mean()))
    sing = np.array(sing); res['single_coordinate_wz_binary'] = dict(n98=float(sing[N98]), rank_of_n98=int((sing > sing[N98]).sum() + 1),
                                                                    top5=[[int(k), float(sing[k])] for k in np.argsort(-sing)[:5]],
                                                                    others_active=[[int(k), float(sing[k])] for k in others],
                                                                    p_perm_vs_127=float((1 + (np.delete(sing, N98) >= sing[N98]).sum()) / 128))
    log(f'  [circuit] single coordinate W/Z binary: n98 {sing[N98]:.4f}, rank {res["single_coordinate_wz_binary"]["rank_of_n98"]}, top5 {res["single_coordinate_wz_binary"]["top5"]}')
    return res


# ------------------------------------------------------------------------------------------------------------ T pass
def t_pass(st, JT, JE, mT, yT, iW, iZ, N98, full, psub):
    from interp.part_core import sub
    t = st.t; N = len(yT)
    share = {k: np.zeros((N, 2, H), np.float32) for k in ['base'] + list(JE)}
    z97 = {k: np.zeros(N, np.float32) for k in ['base'] + list(JE)}; wzl = {k: np.zeros(N, np.float32) for k in ['base'] + list(JE)}
    DEC = ('ang1.1', 'phi0.06'); parts = ['full', 'attn', 'value'] + [f'attn_{lab(bh)}' for bh in HEADS]
    dec = {e: {p: dict(z97=np.zeros(N, np.float32), wz=np.zeros(N, np.float32)) for p in parts} for e in DEC}
    md = 0.0
    for idx in batches(JT, 256):
        Xb, pb, Lm, _ = st.run_X(sub(JT, idx))
        with t.no_grad():
            zb, Lb, Rb = st.cls(Xb, pb, rec=True)
        md = max(md, float((Lb - Lm).abs().max()))
        share['base'][idx] = t.stack(Rb['A'], 1)[..., 0].cpu().numpy(); z97['base'][idx] = zb[:, N98].cpu().numpy()
        wzl['base'][idx] = (Lb[:, iW] - Lb[:, iZ]).cpu().numpy()
        for e, J in JE.items():
            Xe, pe, Lme, _ = st.run_X(sub(J, idx)); assert bool((pe == pb).all())
            with t.no_grad():
                ze, Le, Re = st.cls(Xe, pe, rec=True)
                md = max(md, float((Le - Lme).abs().max()))
                share[e][idx] = t.stack(Re['A'], 1)[..., 0].cpu().numpy(); z97[e][idx] = ze[:, N98].cpu().numpy()
                wzl[e][idx] = (Le[:, iW] - Le[:, iZ]).cpu().numpy()
                if e in DEC:
                    allh = t.ones(H, dtype=t.bool, device=st.dev)
                    runs = dict(full=(Xe, pe, {}), attn=(Xb, pb, {('attn', b): (Re['A'][b], allh) for b in range(2)}),
                                value=(Xe, pe, {('attn', b): (Rb['A'][b], allh) for b in range(2)}))
                    for bh in HEADS:
                        sel = t.tensor([h == bh[1] for h in range(H)], device=st.dev)
                        runs[f'attn_{lab(bh)}'] = (Xb, pb, {('attn', bh[0]): (Re['A'][bh[0]], sel)})
                    for p, (X, pd, ctl) in runs.items():
                        z, L, _ = st.cls(X, pd, ctl)
                        dec[e][p]['z97'][idx] = (z[:, N98] - zb[:, N98]).cpu().numpy()
                        dec[e][p]['wz'][idx] = ((L[:, iW] - L[:, iZ]) - (Lb[:, iW] - Lb[:, iZ])).cpu().numpy()
    log(f'  [{st.name}] T pass done: re-implementation max |dlogit| {md:.2e}')
    # particle-block far-side patch on T[psub]
    pp = None
    if full:
        rng = np.random.default_rng(SEED + 7); allp = [(i, h) for i in range(8) for h in range(H)]
        sets = dict(far_side_h4_h8=FAR_P, all64=allp)
        for k in range(NPSET):
            sets[f'random16_{k}'] = [allp[i] for i in rng.choice(64, 16, replace=False)]
        Js = sub(JT, psub); Jes = {e: sub(JE[e], psub) for e in DEC}; n = len(psub)
        out = {e: {s: dict(z97=np.zeros(n, np.float32), wz=np.zeros(n, np.float32)) for s in ['edit'] + list(sets)} for e in DEC}
        st.recon = 0.0
        for idx in batches(Js, 96):
            st.pm = 'record'; _, _, Lb, zb = st.run_X(sub(Js, idx))
            for e in DEC:
                st.pm = None; _, _, Le, ze = st.run_X(sub(Jes[e], idx))
                out[e]['edit']['z97'][idx] = (ze[:, N98] - zb[:, N98]).cpu().numpy(); out[e]['edit']['wz'][idx] = ((Le[:, iW] - Le[:, iZ]) - (Lb[:, iW] - Lb[:, iZ])).cpu().numpy()
                for s, hs in sets.items():
                    st.pm = hs; _, _, Lp, zp = st.run_X(sub(Jes[e], idx))
                    out[e][s]['z97'][idx] = (zp[:, N98] - zb[:, N98]).cpu().numpy(); out[e][s]['wz'][idx] = ((Lp[:, iW] - Lp[:, iZ]) - (Lb[:, iW] - Lb[:, iZ])).cpu().numpy()
            st.pm = None; st.rec = {}
        pp = dict(out=out, sets={s: [f'p{i + 1}h{h + 1}' for i, h in hs] for s, hs in sets.items()}, recon_max_err=st.recon)
        log(f'  [{st.name}] particle-block patch done: per-head reconstruction max error {st.recon:.2e}')
    return dict(share=share, z97=z97, wzl=wzl, dec=dec, reimpl_max_dlogit=md, pp=pp)


def logistic(m, lo, hi, m50, w):
    return lo + (hi - lo) / (1.0 + np.exp(np.clip((m - m50) / w, -50, 50)))


def fit_logistic(m, s):
    from scipy.optimize import curve_fit
    p, _ = curve_fit(logistic, m, s, p0=[0.05, 0.9, 85.0, 5.0], bounds=([-0.5, 0.0, 40.0, 0.3], [1.0, 1.5, 160.0, 60.0]), maxfev=20000)
    return p


def threshold_analysis(TP, mT, yT, iW, iZ, rng, sd97, edits):
    S = TP['share']['base']; N = len(mT); Wb = boot_W(N, NB, rng)
    edges = np.arange(40, 205, 5.0); cen = 0.5 * (edges[1:] + edges[:-1]); res = dict(bins=cen.tolist(), heads={})
    inb = [(mT >= lo) & (mT < hi) for lo, hi in zip(edges[:-1], edges[1:])]; cnt = np.array([b.sum() for b in inb])
    res['counts'] = cnt.tolist(); okspan = (cen > 50) & (cen < 130) & (cnt >= 50)
    spans, ps = {}, []
    for bh in HEADS:
        s = S[:, bh[0], bh[1]].astype(np.float64); mean = np.full(len(cen), np.nan); lo95 = mean.copy(); hi95 = mean.copy(); bm = np.full((NB, len(cen)), np.nan)
        for j, b in enumerate(inb):
            if cnt[j] < 20: continue
            bf = b.astype(np.float32); mean[j] = s[b].mean(); bm[:, j] = (Wb @ (s * bf)) / np.maximum(Wb @ bf, 1)
            lo95[j], hi95[j] = np.percentile(bm[:, j], [2.5, 97.5])
        span = float(np.nanmax(mean[okspan]) - np.nanmin(mean[okspan])); bspan = np.nanmax(bm[:, okspan], 1) - np.nanmin(bm[:, okspan], 1)
        p = float(max(1.0 / NB, (bspan <= 0.20).mean())); ps.append(p)
        res['heads'][lab(bh)] = dict(mean=mean.tolist(), lo=lo95.tolist(), hi=hi95.tolist(), span_50_130=span, span_ci=ci(bspan), p_span_le_20pt=p,
                                     mean_W=float(s[yT == iW].mean()), mean_Z=float(s[yT == iZ].mean()))
    for bh, pa in zip(HEADS, holm(ps)): res['heads'][lab(bh)]['p_span_le_20pt_holm'] = pa
    # logistic fit for b2h8 (and the next-largest-span head), per jet on 50-140 GeV; also W only and Z only
    rank = sorted(HEADS, key=lambda bh: -res['heads'][lab(bh)]['span_50_130']); res['rank_by_span'] = [lab(bh) for bh in rank]
    fits = {}
    for bh in [B2H8] + [r for r in rank if r != B2H8][:1]:
        for sub_, sel0 in (('WZ', np.ones(N, bool)), ('W', yT == iW), ('Z', yT == iZ)):
            sel = sel0 & (mT >= 50) & (mT <= 140); m = mT[sel].astype(np.float64); s = S[sel, bh[0], bh[1]].astype(np.float64)
            try:
                p = fit_logistic(m, s); bp = []
                for _ in range(NFIT):
                    ii = rng.integers(0, len(m), len(m))
                    try: bp.append(fit_logistic(m[ii], s[ii]))
                    except Exception: pass
                bp = np.array(bp)
                fits[f'{lab(bh)}_{sub_}'] = dict(n=int(sel.sum()), lo=float(p[0]), hi=float(p[1]), m50=float(p[2]), width=float(p[3]),
                                                 m50_ci=ci(bp[:, 2]), width_ci=ci(bp[:, 3]), lo_ci=ci(bp[:, 0]), hi_ci=ci(bp[:, 1]), n_boot_ok=len(bp))
            except Exception as ex:
                fits[f'{lab(bh)}_{sub_}'] = dict(error=str(ex))
            log(f'  logistic {lab(bh)} {sub_}: {fits[f"{lab(bh)}_{sub_}"]}')
    res['logistic'] = fits
    # edits: delta share per head (W/Z jets), with the observational curve's prediction for b2h8
    ed = {}
    for e, me in edits.items():
        row = {}; ps = []
        for bh in HEADS:
            d = (TP['share'][e][:, bh[0], bh[1]] - S[:, bh[0], bh[1]]).astype(np.float64); m_, bm = mean_stats(Wb, d); p = p_two(bm)
            row[lab(bh)] = dict(mean=m_, ci=ci(bm), p0=p, W=float(d[yT == iW].mean()), Z=float(d[yT == iZ].mean())); ps.append(p)
        for bh, pa in zip(HEADS, holm(ps)): row[lab(bh)]['p0_holm'] = pa
        f = fits.get(f'{lab(B2H8)}_WZ', {})
        if 'm50' in f:
            pr = logistic(me, f['lo'], f['hi'], f['m50'], f['width']) - logistic(mT, f['lo'], f['hi'], f['m50'], f['width'])
            row['b2h8_curve_prediction'] = dict(mean=float(pr.mean()), W=float(pr[yT == iW].mean()), Z=float(pr[yT == iZ].mean()))
        dz = (TP['z97'][e] - TP['z97']['base']) / sd97; dwz = TP['wzl'][e] - TP['wzl']['base']; dm = (me - mT) / 10.0
        row['true_dm_mean_GeV'] = float((me - mT).mean())
        row['dz97_mean'] = dict(all=float(dz.mean()), W=float(dz[yT == iW].mean()), Z=float(dz[yT == iZ].mean()))
        row['dwz_mean'] = dict(all=float(dwz.mean()), W=float(dwz[yT == iW].mean()), Z=float(dwz[yT == iZ].mean()))
        if np.abs(dm).mean() > 1e-3:
            s, bs = slope_stats(Wb, dz.astype(np.float64), dm); row['dz97_per_10GeV'] = dict(slope=s, ci=ci(bs))
            s, bs = slope_stats(Wb, dwz.astype(np.float64), dm); row['dwz_per_10GeV'] = dict(slope=s, ci=ci(bs))
        others = max((abs(row[lab(bh)]['mean']), lab(bh)) for bh in HEADS if bh != B2H8)
        row['largest_other_head'] = dict(head=others[1], abs_mean=others[0])
        ed[e] = row
        log(f'  edit {e}: true dm {row["true_dm_mean_GeV"]:+.2f} GeV | b2h8 dshare {row["b2h8"]["mean"]:+.4f} {np.round(row["b2h8"]["ci"], 4).tolist()} '
            f'(curve prediction {row.get("b2h8_curve_prediction", {}).get("mean")}) | largest other {others[1]} {others[0]:.4f} | dz97 W {row["dz97_mean"]["W"]:+.3f} Z {row["dz97_mean"]["Z"]:+.3f}')
    res['edits'] = ed
    return res


def decomposition_analysis(TP, mT, yT, iW, iZ, rng, sd97):
    sel = (mT >= 70) & (mT <= 110); n = int(sel.sum()); Wb = boot_W(n, NB, rng); out = dict(n=n, window='W/Z jets of T with base mass 70-110 GeV')
    for e, D in TP['dec'].items():
        row = {}
        for key, sc in (('z97', sd97), ('wz', 1.0)):
            f = D['full'][key][sel] / sc; a = D['attn'][key][sel] / sc; v = D['value'][key][sel] / sc; it = f - a - v
            r = {}
            for nm, x in (('total', f), ('attention', a), ('value', v), ('interaction', it)):
                m_, bm = mean_stats(Wb, x.astype(np.float64)); r[nm] = dict(mean=m_, ci=ci(bm))
            bt = (Wb @ f.astype(np.float64)) / Wb.sum(1)
            r['attention_share'] = dict(value=r['attention']['mean'] / r['total']['mean'], ci=ci(((Wb @ a.astype(np.float64)) / Wb.sum(1)) / bt))
            heads = {}
            for bh in HEADS:
                x = D[f'attn_{lab(bh)}'][key][sel] / sc; m_, bm = mean_stats(Wb, x.astype(np.float64))
                heads[lab(bh)] = dict(mean=m_, ci=ci(bm), share_of_total=m_ / r['total']['mean'], share_ci=ci(bm / bt), p0=p_two(bm))
            for bh, pa in zip(HEADS, holm([heads[lab(b)]['p0'] for b in HEADS])): heads[lab(bh)]['p0_holm'] = pa
            r['per_head_attention'] = heads; r['sum_of_per_head_attention'] = float(sum(h['mean'] for h in heads.values()))
            row[key] = r
        out[e] = row
        z = row['z97']
        log(f'  G21b {e}: dz97 total {z["total"]["mean"]:+.4f} = attention {z["attention"]["mean"]:+.4f} + value {z["value"]["mean"]:+.4f} + '
            f'interaction {z["interaction"]["mean"]:+.4f}; b2h8 attention part {z["per_head_attention"]["b2h8"]["mean"]:+.4f} '
            f'(share {z["per_head_attention"]["b2h8"]["share_of_total"]:+.3f}); top heads '
            + ', '.join(f'{k} {v["mean"]:+.4f}' for k, v in sorted(z['per_head_attention'].items(), key=lambda kv: -abs(kv[1]['mean']))[:4]))
    return out


def ppatch_analysis(pp, yp, iW, iZ, rng, sd97):
    if pp is None: return None
    n = len(yp); Wb = boot_W(n, NB, rng); res = dict(sets=pp['sets'], recon_max_err=pp['recon_max_err'])
    for e, O in pp['out'].items():
        row = {}
        for key, sc in (('z97', sd97), ('wz', 1.0)):
            for cn, c in (('W', iW), ('Z', iZ)):
                s = yp == c; Wc = Wb[:, s]
                ed = O['edit'][key][s] / sc; bt = (Wc @ ed) / Wc.sum(1); r = dict(edit=float(ed.mean()), edit_ci=ci(bt))
                for st_, D in O.items():
                    if st_ == 'edit': continue
                    x = D[key][s] / sc; rem = 1 - x.mean() / ed.mean(); brem = 1 - ((Wc @ x) / Wc.sum(1)) / bt
                    r[st_] = dict(after_patch=float(x.mean()), removed_share=float(rem), removed_ci=ci(brem))
                rr = np.array([r[f'random16_{k}']['removed_share'] for k in range(NPSET)])
                r['random16_removed'] = dict(mean=float(rr.mean()), min=float(rr.min()), max=float(rr.max()), values=rr.tolist())
                r['p_far_vs_random16'] = float((1 + (rr >= r['far_side_h4_h8']['removed_share']).sum()) / (1 + NPSET))
                for k in range(NPSET): del r[f'random16_{k}']
                row[f'{key}_{cn}'] = r
        res[e] = row
        r = row['z97_W']
        log(f'  particle far-side patch {e}: dz97 on W {r["edit"]:+.4f}; far side removes {r["far_side_h4_h8"]["removed_share"]:.3f} '
            f'{np.round(r["far_side_h4_h8"]["removed_ci"], 3).tolist()}, all 64 {r["all64"]["removed_share"]:.3f}, random 16 mean {r["random16_removed"]["mean"]:.3f} '
            f'max {r["random16_removed"]["max"]:.3f}, p {r["p_far_vs_random16"]:.3f}')
    return res


def all_class_span(D, mass, y):
    edges = np.arange(40, 245, 10.0); out = {}
    for bh in HEADS:
        s = D['share'][:, bh[0], bh[1]]; mm = [s[(mass >= lo) & (mass < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:]) if ((mass >= lo) & (mass < hi)).sum() >= 50]
        out[lab(bh)] = float(max(mm) - min(mm))
    return out


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items() if not str(k).startswith('_')}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.ndarray): return clean(o.tolist())
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, float) and not math.isfinite(o): return None
    return o


# ------------------------------------------------------------------------------------------------------------ main
def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from interp.part_core import kin, build, jmass, sub, cat, sample, CL, N98, PRE
    import torch
    OUT.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(SEED)
    iW, iZ = CL.index('Wqq'), CL.index('Zqq')
    res = dict(seed=SEED, smoke=SMOKE, per_class_S=PER, per_class_T=PER_T, NB=NB, NDIR=NDIR, NRAND=NRAND, NPSET=NPSET, val_dir=str(VAL),
               label_map=dict(class_heads='G-series bXhY = cls_blocks[X-1], head Y-1; b2h8 = (1, 7)', neuron='G-series nK = z index K-1; n98 = z[97]',
                              particle_heads='G101 far-side heads h4, h8 = head index 3, 7 of each particle block (labelled p<block>h<head>, 1-based)'),
               models={})
    # samples
    parts, rowsS = [], {}
    for c in CL:
        J, rows, Nf = load_rows(VAL / f'{PRE[c]}_120.root', PER, rng); parts.append(J); rowsS[c] = dict(file=f'{PRE[c]}_120.root', entries=Nf, first_rows=rows[:5].tolist())
        assert np.all(J['label'] == CL.index(c)), c
    J0 = cat(parts); y = J0['label'].astype(int); JS = build(J0, *kin(J0)); mS = jmass(JS); del parts, J0
    log(f'S: {len(y)} jets ({PER}/class, random rows of <sample>_120.root); mass median W {np.median(mS[y == iW]):.1f} Z {np.median(mS[y == iZ]):.1f} GeV')
    tp = []
    for c in ('Wqq', 'Zqq'):
        J, rows, Nf = load_rows(VAL / f'{PRE[c]}_121.root', PER_T, rng); tp.append(J); rowsS[f'T_{c}'] = dict(file=f'{PRE[c]}_121.root', entries=Nf, first_rows=rows[:5].tolist())
    T0j = cat(tp); yT = T0j['label'].astype(int); pt, de, dp, ms = kin(T0j); real = pt > 0
    JT = build(T0j, pt, de, dp, ms); mT = jmass(JT)
    JE = {'ang0.9': build(T0j, pt, de * 0.9, dp * 0.9, ms), 'ang1.1': build(T0j, pt, de * 1.1, dp * 1.1, ms), 'phi0.06': build(T0j, pt, de, dp + 0.06 * real, ms)}
    mE = {e: jmass(J) for e, J in JE.items()}; del T0j, tp, pt, de, dp, ms, real
    res['samples'] = rowsS; res['mass_T'] = dict(W_median=float(np.median(mT[yT == iW])), Z_median=float(np.median(mT[yT == iZ])))
    res['edit_truth'] = {e: dict(dm_mean=float((m - mT).mean()), dlogm_mean=float(np.mean(np.log(np.maximum(m, 1)) - np.log(np.maximum(mT, 1)))),
                                 max_abs_dm=float(np.abs(m - mT).max()), mask_equal=bool(np.array_equal(JE[e]['mask'], JT['mask']))) for e, m in mE.items()}
    log(f'T: {len(yT)} W/Z jets; edit truth {res["edit_truth"]}')
    psub = np.arange(0, len(yT), 5)
    for name in ('full', 'kin'):
        st = Stage(name); full = name == 'full'; M = {}; t1 = time.time()
        # P1 sample check
        JP = sample(100 if SMOKE else 1000, 3000); JPb = build(JP, *kin(JP)); RP = st.net.run(JPb)
        M['P1_sample_acc'] = float((RP['logits'].argmax(1) == JP['label']).mean()); del JP, JPb, RP
        D = base_pass(st, JS, mS)
        c0 = D['L'].argmax(1); M['S_acc10'] = float((c0 == y).mean())
        wz = np.isin(y, [iW, iZ]); M['S_wz_binary_acc'] = float((np.where(D['L'][:, iW] > D['L'][:, iZ], iW, iZ)[wz] == y[wz]).mean())
        M['reimpl'] = dict(max_dlogit=D['reimpl_max_dlogit'], calls_changed=D['reimpl_calls_changed'])
        log(f'ParT_{name}: P1-sample acc {M["P1_sample_acc"]:.4f}; S acc {M["S_acc10"]:.4f}; S W/Z binary {M["S_wz_binary_acc"]:.4f}; re-implementation max |dlogit| '
            f'{D["reimpl_max_dlogit"]:.2e}, calls changed {D["reimpl_calls_changed"]}')
        M['all_class_span_40_240'] = all_class_span(D, mS, y)
        M['S_share_by_class'] = {lab(bh): {CL[c]: float(D['share'][y == c, bh[0], bh[1]].mean()) for c in range(10)} for bh in HEADS}
        log(f'  all-class span (S, 10 GeV bins 40-240): ' + ', '.join(f'{k} {v:.2f}' for k, v in sorted(M['all_class_span_40_240'].items(), key=lambda kv: -kv[1])[:5]))
        sd97 = float(D['z'][:, N98].std())
        TP = t_pass(st, JT, JE, mT, yT, iW, iZ, N98, full, psub)
        M['threshold'] = threshold_analysis(TP, mT, yT, iW, iZ, np.random.default_rng(SEED + 3), sd97, mE)
        M['decomposition'] = decomposition_analysis(TP, mT, yT, iW, iZ, np.random.default_rng(SEED + 4), sd97)
        M['reimpl_T_max_dlogit'] = TP['reimpl_max_dlogit']
        M['particle_far_side_patch'] = ppatch_analysis(TP['pp'], yT[psub], iW, iZ, np.random.default_rng(SEED + 5), sd97)
        del TP
        E11 = M['threshold']['edits']['ang1.1']; ref = dict(z97=E11['dz97_per_10GeV']['slope'], wz=E11['dwz_per_10GeV']['slope'])
        HP = heads_part(st, D, y, mS, np.random.default_rng(SEED + 1), iW, iZ, N98, full, ref); M['heads'] = HP
        if full: M['circuits'] = circuits_part(st, D, y, np.random.default_rng(SEED + 2), iW, iZ, N98, HP['_named'])
        for bt in D['bat']: bt['X'] = None
        del D
        res['models'][name] = M; del st; torch.cuda.empty_cache() if torch.cuda.is_available() else None
        log(f'ParT_{name} done ({time.time() - t1:.0f} s)')
        json.dump(clean(res), open(FN, 'w'), indent=1)
    json.dump(clean(res), open(FN, 'w'), indent=1); log(f'wrote {FN}')
    try: plot(clean(res))
    except Exception as ex: log(f'plot failed: {ex!r}')
    log('P3 DONE')


# ------------------------------------------------------------------------------------------------------------ figure
def plot(res=None):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    if res is None: res = json.load(open(FN))
    M = res['models']['full']; th = M['threshold']; cen = np.array(th['bins'])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), dpi=150)
    a = ax[0]
    for k, v in th['heads'].items():
        if k == 'b2h8': continue
        a.plot(cen, np.array(v['mean'], float) * 100, color='#bbbbbb', lw=0.8)
    v = th['heads']['b2h8']; mn = np.array(v['mean'], float) * 100
    a.fill_between(cen, np.array(v['lo'], float) * 100, np.array(v['hi'], float) * 100, color='#c0392b', alpha=0.25, lw=0)
    a.plot(cen, mn, color='#c0392b', lw=2, label='head b2h8 (95 % band)')
    f = th['logistic'].get('b2h8_WZ', {})
    if 'm50' in f:
        mm = np.linspace(50, 140, 200); a.plot(mm, logistic(mm, f['lo'], f['hi'], f['m50'], f['width']) * 100, 'k--', lw=1,
                                               label=f'logistic fit, m50 = {f["m50"]:.1f} GeV [{f["m50_ci"][0]:.1f}, {f["m50_ci"][1]:.1f}]')
    a.plot([], [], color='#bbbbbb', label='other 15 class-attention heads')
    a.set_xlim(40, 200); a.set_ylim(0, 100); a.set_xlabel('jet mass [GeV]'); a.set_ylabel('attention on the class token [%]')
    a.set_title('(a) b2h8 class-token share vs jet mass\n(W and Z jets)', fontsize=10); a.legend(fontsize=7, loc='upper right')
    a = ax[1]; H_ = M['heads']['heads']; ks = list(H_.keys())
    sh = np.array([H_[k]['resample']['cal_z97'] for k in ks]); lo = np.array([H_[k]['resample']['cal_z97_ci'][0] for k in ks]); hi = np.array([H_[k]['resample']['cal_z97_ci'][1] for k in ks])
    allc = M['heads']['groups']['all16']['resample']['cal_z97']
    col = ['#c0392b' if k == 'b2h8' else ('#2c7fb8' if k in M['heads']['named_heads_F'] else '#888888') for k in ks]
    a.bar(range(len(ks)), sh * 100, color=col); a.errorbar(range(len(ks)), sh * 100, yerr=[(sh - lo) * 100, (hi - sh) * 100], fmt='none', ecolor='k', lw=0.8)
    a.axhline(allc * 100, color='k', ls='--', lw=0.8, label=f'all 16 heads resampled together ({allc * 100:.0f} %)'); a.legend(fontsize=7, loc='upper right')
    a.set_xticks(range(len(ks))); a.set_xticklabels(ks, rotation=90, fontsize=7)
    a.set_ylabel('n98 change per 10 GeV, % of a real mass edit'); a.set_title('(b) n98 response when one head is taken from a jet of other mass\n(same class; relative to scaling all angles by 1.1)', fontsize=10)
    a = ax[2]; C_ = M['circuits']; CV = C_['variants']['uniform_S']; names = [k for k in ('V_a_heads_n98_strict', 'V_b_heads_32mlp_n98', 'V_c_heads_allmlp_n98', 'V_d_heads_only') if k in CV]
    short = {'V_a_heads_n98_strict': 'heads + n98\n(MLP resampled)', 'V_b_heads_32mlp_n98': 'heads + 32 MLP\n+ n98', 'V_c_heads_allmlp_n98': 'heads + all MLP\n+ n98', 'V_d_heads_only': 'heads only\n(rest intact)'}
    for i, k in enumerate(names):
        e = CV[k]; e2 = C_['variants']['WZ_only'][k]
        a.scatter([i + 0.25], [e2['wz_binary_acc'] * 100], s=40, facecolors='none', edgecolors='#c0392b', zorder=3)
        if 'random_wz_binary_acc' in e:
            rv = np.array(e['random_wz_binary_acc']['values']) * 100; a.scatter(i + np.random.default_rng(0).uniform(-0.15, 0.15, len(rv)), rv, s=5, color='#999999')
        a.scatter([i], [e['wz_binary_acc'] * 100], s=50, color='#c0392b', zorder=3)
    a.axhline(C_['unedited']['wz_binary_acc'] * 100, color='k', ls='--', lw=0.8, label='unedited ParT_full'); a.axhline(50, color='k', ls=':', lw=0.8)
    a.scatter([], [], color='#c0392b', label='literal circuit (donors: all classes)'); a.scatter([], [], facecolors='none', edgecolors='#c0392b', label='literal circuit (donors: W/Z jets)'); a.scatter([], [], s=5, color='#999999', label='random circuits, equal size')
    a.set_xticks(range(len(names))); a.set_xticklabels([short[k] for k in names], fontsize=7); a.set_ylabel('W vs Z accuracy on true W/Z jets [%]')
    a.set_ylim(45, 88); a.set_title('(c) circuit {b2h8, b1h3, n98} kept alone (rest resampled)', fontsize=10); a.legend(fontsize=7, loc='upper left')
    fig.suptitle('ParT_full: head b2h8 switches at 87 GeV, but little of the mass reaches neuron n98 through it (P3)', fontsize=11)
    fig.tight_layout(); p = OUT / 'p3_wz_circuit.png'; fig.savefig(p, bbox_inches='tight'); log(f'wrote {p}')


if __name__ == '__main__':
    if '--plot' in sys.argv: plot()
    else: main()
