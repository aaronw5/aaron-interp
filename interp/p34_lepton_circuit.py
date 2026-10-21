"""P34 (loop C, round C8; claim C15) — lepton route follow-up: flag edits on a fresh sample and the literal lepton circuit.

Sample: JetClass test_20M <sample>_116.root (all 10 classes), numpy default_rng(3416), random rows (never first-k):
  NE rows per class = evaluation sample; NREF further rows per class (disjoint) = reference set for the mean ablation.
  Every jet passes through part_core.build; edited jets are compared with base = rebuild(BL) (same number of build passes).
  Reference check: unedited 10-class accuracy on the P1 sample (part_core.sample(1000, 3000); ParT_full 0.8593 in P32).

Part 1 — flag edits (ParT_full, ParT_kinpid, ParT_kin, PN_full, PN_kinpid) on H->lvqq' (Hqql) jets with a lepton candidate
  (highest-pT particle with isElectron or isMuon) and a pT-matched charged hadron (random charged hadron with pT within x1.5,
  else nearest in pT; as P32):
  e1   lepton flag removed: isElectron = isMuon = 0, isChargedHadron = 1 (four-momentum, charge, d0/dz unchanged).
  e2n  control, same-size one-hot change on a matched-pT token with no lepton flag involved: matched charged hadron -> neutral hadron.
  e0   no-op check: isElectron = isMuon = 0 written on the neutral hadron nearest in pT to the lepton (already 0) -> must give 0.
  e1x  electron <-> muon swap on the lepton (flag identity vs flag presence).
  Metrics: share of calls (argmax) changed, Delta log p(Hqql), Delta log-odds(Hqql); McNemar (paired, exact) for e1 vs e2n and
  e1 vs e1x call changes; 95 % bootstrap over jets (NB resamples).
  ParT internals (all three ParT models): lepton-key knock-out (attention logit -> -1e9; renormalises the head over the other
  keys) and lepton value-zeroing (the lepton's value vector set to 0 in that head; weights unchanged) in the dominant class head
  (ParT_full b2h1, ParT_kinpid/kin b2h4), vs the same on the matched hadron; class16 / particle64 / all80 joint knock-outs;
  dominant + k random heads vs the same k random heads (k = 1, 3, 7, 15, 31; NRKO sets each; paired per set).

Part 2 — literal circuit (ParT_full, ParT_kinpid). Kept: the particle embedding (input encoding; every input reaches the network
  only through it), the pair embedding (only its bias into kept heads matters, the other heads' outputs are replaced), the kept
  heads, all LayerNorms, the class token parameter and the final linear layer. Mean-ablated (replaced by the per-unit mean over the
  reference set, NREF jets per class, unablated model): every other head's output (per head, 128-dim contribution after the output
  projection, mean over real query tokens; class heads over the class token) and the output of every MLP (fc2 output of the 8
  particle blocks and the 2 class blocks; mean over real tokens / the class token).
  Circuits: top-k heads by P32's single-head lepton-key knock-out among its Holm-significant heads (k = 1, 3, 5, 10), mixed,
  class-block only and particle-block only; the same top-k with MLPs kept (P32's head-only ablation); all 80 heads with MLPs
  mean-ablated; nothing kept (all heads + MLPs at mean -> constant output, check).
  Controls: k = 1: all 80 single-head circuits (exhaustive); k = 3, 5, 10: NR random head sets with the same number of class and
  particle heads (empirical p = (1 + #random >= selected) / (1 + NR)); class-only / particle-only: NR2 random sets of that block type.
  Random sets are evaluated on a class-balanced subsample (NSUB per class of the evaluation sample), the selected circuits on both.
  Metrics: Hqql recall, class-balanced accuracy (mean of the 10 per-class recalls), 10-class accuracy, 10x10 confusion, share of
  t->blv jets called Hqql, AUC of the Hqql log-odds for Hqql vs all others and vs t->blv; e1 / e2n applied under the circuit
  (does the circuit's Hqql call depend on the flag).
  ./sync.sh && ./part_job.sh c-p34-run 'python -m interp.p34_lepton_circuit --smoke && python -m interp.p34_lepton_circuit'
  -> $AI_RESULTS/part/p34_lepton_circuit[_smoke].json; --plot -> $AI_RESULTS/part/p34_lepton_circuit.png
"""
import json, math, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import interp.p32_lepton_route as P   # noqa: E402  (parses --smoke / --plot)

SMOKE = P.SMOKE
SEED = 3416
FILE = '116'
NE = 60 if SMOKE else 1500
NREF = 20 if SMOKE else 300
NSUB = 30 if SMOKE else 300
NB = 100 if SMOKE else 1000
NR = 4 if SMOKE else 100
NR2 = 3 if SMOKE else 30
NRKO = 3 if SMOKE else 20
KS_KO = (1, 3, 7, 15, 31)
HQQL, TBL, H, C, HD = 5, 9, 8, 128, 16
FN = P.OUTD / f'p34_lepton_circuit{P.TAG}.json'
log = P.log
HEADS = P.HEADS


def lab2ih(s):
    b, h = s[1:].split('h'); return (int(b) - 1, int(h) - 1) if s[0] == 'p' else (int(b) + 7, int(h) - 1)


# rankings: P32 (results/part/p32_lepton_route.json) single-head lepton-key knock-out, Holm-significant heads ordered by
# effect (lepton KO minus matched-hadron KO on Hqql log-odds); class-only / particle-only = the same list filtered
RANK = {
    'ParT_full': dict(all=['b2h1', 'p7h8', 'p8h6', 'p8h2', 'b2h6', 'p3h2', 'p6h7', 'p5h4', 'p7h7', 'b2h8'],
                      cls=['b2h1', 'b2h6', 'b2h8', 'b1h7', 'b1h4', 'b2h2', 'b2h3', 'b1h1', 'b2h5'],
                      par=['p7h8', 'p8h6', 'p8h2', 'p3h2', 'p6h7', 'p5h4', 'p7h7', 'p6h1', 'p2h2', 'p8h1']),
    'ParT_kinpid': dict(all=['b2h4', 'p8h5', 'p4h5', 'p3h5', 'p8h4', 'b1h8', 'p7h2', 'p4h8', 'p2h4', 'p2h5'],
                        cls=['b2h4', 'b1h8', 'b1h1', 'b1h5', 'b1h3'],
                        par=['p8h5', 'p4h5', 'p3h5', 'p8h4', 'p7h2', 'p4h8', 'p2h4', 'p2h5', 'p5h4', 'p4h1'])}
DOM = {'ParT_full': (9, 0), 'ParT_kinpid': (9, 3), 'ParT_kin': (9, 3)}
KS = (1, 3, 5, 10)
PMLP, CMLP = set(range(8)), {8, 9}


# ------------------------------------------------------------------------------------------------ ParT with head + MLP controls
class Run:
    """ParT with per-head controls (key mask, value zeroing, mean replacement), MLP mean replacement and mean recording.
    Attention modules 0-7 = particle blocks, 8-9 = class blocks. ctl per batch:
      kmask {i: (n, H, S) additive}; vzero {i: (n, H, S) multiplicative on values}; mean {i: (sel (H,) bool, M (H, C))};
      mlp set(i) of blocks whose MLP output is replaced by its mean; rec True (record head and MLP means)."""

    def __init__(self, name, feat):
        import torch
        from interp.part_core import ParTNetwork
        self.t, self.name, self.feat = torch, name, feat
        self.net = ParTNetwork(feat); self.m = self.net.model; self.dev = self.net.device; self.m.eval()
        self.ctl = {}; self.qmask = None; self.recon = 0.0
        z = lambda: {i: torch.zeros(H, C, dtype=torch.float64, device=self.dev) for i in range(10)}
        self.hsum = z(); self.hcnt = {i: 0.0 for i in range(10)}
        self.msum = {i: torch.zeros(C, dtype=torch.float64, device=self.dev) for i in range(10)}; self.mcnt = {i: 0.0 for i in range(10)}
        self.HM = None; self.MM = None
        blocks = list(self.m.blocks) + list(self.m.cls_blocks)
        for i, b in enumerate(blocks):
            self._wrap(i, b.attn); b.fc2.register_forward_hook(lambda mod, inp, out, i=i: self._mlp(i, out))

    def _mlp(self, i, out):
        c = self.ctl; t = self.t
        if c.get('rec'):
            if i < 8:
                q = self.qmask                                                     # (P, n) real tokens
                self.msum[i] += (out * q[..., None]).sum((0, 1)).double(); self.mcnt[i] += float(q.sum())
            else:
                self.msum[i] += out.sum((0, 1)).double(); self.mcnt[i] += float(out.shape[0] * out.shape[1])
        if i in c.get('mlp', ()):
            return t.zeros_like(out) + self.MM[i]
        return None

    def _wrap(self, i, attn):
        orig, S, t = attn.forward, self, self.t

        def f(query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True):
            c = S.ctl; km = c.get('kmask', {}).get(i); vz = c.get('vzero', {}).get(i); mean = c.get('mean', {}).get(i); rec = c.get('rec')
            if km is None and vz is None and mean is None and not rec:
                return orig(query, key, value, key_padding_mask=key_padding_mask, need_weights=False, attn_mask=attn_mask)
            Lq, n, _ = query.shape; Sk = key.shape[0]
            if km is not None:
                am = km[:, :, None, :Sk].expand(n, H, Lq, Sk).reshape(n * H, Lq, Sk)
                attn_mask = am if attn_mask is None else attn_mask + am
            out, w = orig(query, key, value, key_padding_mask=key_padding_mask, need_weights=True, attn_mask=attn_mask, average_attn_weights=False)
            realq = (~key_padding_mask).float() if i < 8 else t.ones(n, 1, device=query.device)
            v = (value @ attn.in_proj_weight[2 * C:].T + attn.in_proj_bias[2 * C:]).view(Sk, n, H, HD).permute(1, 2, 0, 3)   # (n, H, Sk, HD)
            Wo = attn.out_proj.weight.view(C, H, HD)
            ch = t.einsum('nhqd,chd->nhqc', w @ v, Wo)                                                                    # (n, H, Lq, C)
            if rec:
                S.recon = max(S.recon, float((ch.sum(1) + attn.out_proj.bias - out.permute(1, 0, 2)).abs().max()))
                S.hsum[i] += (ch * realq[:, None, :, None]).sum((0, 2)).double(); S.hcnt[i] += float(realq.sum())
            d = 0
            if vz is not None:
                ch2 = t.einsum('nhqd,chd->nhqc', w @ (v * vz[:, :, :Sk, None]), Wo); d = d + (ch2 - ch).sum(1); ch = ch2
            if mean is not None:
                sel, M = mean; d = d + ((M[None, :, None, :] - ch) * sel[None, :, None, None]).sum(1)
            if vz is not None or mean is not None: out = out + d.permute(1, 0, 2)
            return out, None
        attn.forward = f

    def fix_means(self):
        self.HM = {i: (self.hsum[i] / max(self.hcnt[i], 1)).float() for i in range(10)}
        self.MM = {i: (self.msum[i] / max(self.mcnt[i], 1)).float() for i in range(10)}

    def run(self, J, mk=None, bs=P.BS):
        from interp.part_core import sub, inputs
        t = self.t; N = len(J['mask']); order = np.argsort(J['mask'].sum(1), kind='stable'); L = np.zeros((N, 10), np.float32)
        for s in range(0, N, bs):
            idx = order[s:s + bs]; Jb = sub(J, idx); x, v, m = inputs(Jb, self.feat); Pn = x.shape[-1]
            self.ctl = mk(idx, Pn) if mk else {}
            self.qmask = t.from_numpy((Jb['mask'][:, :Pn].T > 0).astype(np.float32)).to(self.dev)
            with t.no_grad(): L[idx] = self.m(*(t.from_numpy(a).to(self.dev) for a in (x, v, m))).float().cpu().numpy()
        self.ctl = {}; return L

    # ---- control builders
    def kmask(self, n, Pn, heads, tok):
        return P.kmask_for(self, n, Pn, heads, tok)

    def vzero(self, n, Pn, heads, tok):
        t = self.t; d = {}; ar = t.arange(n, device=self.dev); tk = t.from_numpy(np.asarray(tok)).to(self.dev).long()
        for (i, h) in heads:
            if i not in d: d[i] = t.ones(n, H, Pn + 1, device=self.dev)
            d[i][ar, h, tk + (0 if i < 8 else 1)] = 0.0
        return d

    def circ_ctl(self, keep, mlp_ablate=True):
        t = self.t; d = {}
        for (i, h) in HEADS:
            if (i, h) in keep: continue
            if i not in d: d[i] = (t.zeros(H, dtype=t.bool, device=self.dev), self.HM[i])
            d[i][0][h] = True
        c = dict(mean=d)
        if mlp_ablate is True: c['mlp'] = set(range(10))
        elif mlp_ablate: c['mlp'] = set(mlp_ablate)          # a set of block indices (0-7 particle, 8-9 class)
        return c


# ------------------------------------------------------------------------------------------------ helpers
def auc(pos, neg):
    from scipy.stats import rankdata
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg): return None
    r = rankdata(np.concatenate([pos, neg])); return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def metr(L, y, tok, W=None, conf=True):
    c = L.argmax(1); e = P.acc_block(L, y, W); lo = P.lodds(L)
    e['hqql_recall'] = e['recall'][HQQL]
    e['tbl_called_hqql'] = float((c[y == TBL] == HQQL).mean())
    e['other_called_hqql'] = float((c[y != HQQL] == HQQL).mean())
    oth = (y != HQQL) & (y != TBL)
    e['other_nontbl_called_hqql_with_lepton'] = float((c[oth & (tok >= 0)] == HQQL).mean()) if (oth & (tok >= 0)).any() else None
    e['other_nontbl_called_hqql_without_lepton'] = float((c[oth & (tok < 0)] == HQQL).mean()) if (oth & (tok < 0)).any() else None
    e['auc_hqql_vs_rest'] = auc(lo[y == HQQL], lo[y != HQQL]); e['auc_hqql_vs_tbl'] = auc(lo[y == HQQL], lo[y == TBL])
    e['pred_share'] = (np.bincount(c, minlength=10) / len(c)).tolist(); e['n_distinct_calls'] = int(len(np.unique(c)))
    if W is not None:
        hq = y == HQQL
        e['hqql_recall_ci'] = P.ci(W[:, hq] @ (c[hq] == HQQL).astype(np.float32) / W[:, hq].sum(1))
        e['balanced_ci'] = P.ci(np.stack([W[:, y == k] @ (c[y == k] == k).astype(np.float32) / W[:, y == k].sum(1) for k in range(10)], 1).mean(1))
        tb = y == TBL; e['tbl_called_hqql_ci'] = P.ci(W[:, tb] @ (c[tb] == HQQL).astype(np.float32) / W[:, tb].sum(1))
    if not conf: e.pop('confusion')
    return e


def emp_p(sel, rand):
    rand = np.asarray([r for r in rand if r is not None], float)
    return float((1 + (rand >= sel).sum()) / (1 + len(rand))) if len(rand) else None


def summ(rv, keys=('hqql_recall', 'balanced', 'acc10', 'tbl_called_hqql', 'auc_hqql_vs_rest', 'auc_hqql_vs_tbl')):
    o = {}
    for k in keys:
        a = np.asarray([r[k] for r in rv if r[k] is not None], float)
        o[k] = dict(mean=float(a.mean()), q05=float(np.percentile(a, 5)), q95=float(np.percentile(a, 95)), min=float(a.min()), max=float(a.max()))
    return o


def flavswap(B, tok, ise):
    E = {k: (v.copy() if np.ndim(v) else v) for k, v in B.items()}; r = np.arange(len(tok))
    for k in ('isElectron', 'isMuon'): E[k][r, tok] = 0
    E['isMuon'][r[ise], tok[ise]] = 1; E['isElectron'][r[~ise], tok[~ise]] = 1
    return P.rebuild(E)


def noop(B, nh):
    """isElectron = isMuon = 0 written on token nh (a neutral hadron, already 0): identity edit through the same pipeline"""
    E = {k: (v.copy() if np.ndim(v) else v) for k, v in B.items()}; r = np.arange(len(nh))
    for k in ('isElectron', 'isMuon'): E[k][r, nh] = 0
    return P.rebuild(E)


def matched_neutral(J, tok):
    from interp.part_core import kin
    pt = kin(J)[0]; nh = (J['isNeutralHadron'] > 0.5) & (J['mask'] > 0); out = np.full(len(tok), -1)
    for j in range(len(tok)):
        c = np.where(nh[j])[0]
        if len(c): out[j] = c[np.argmin(np.abs(np.log(pt[j, c] / pt[j, tok[j]])))]
    return out


# ------------------------------------------------------------------------------------------------ main
def main():
    import torch
    P.add_kinpid()
    from interp.part_core import sub, cat, PRE, CL, sample
    P.OUTD.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(SEED); torch.manual_seed(SEED)
    R = dict(node='P34', seed=SEED, smoke=SMOKE, files=f'test_20M <sample>_{FILE}.root', NE=NE, NREF=NREF, NSUB=NSUB, NB=NB, NR=NR, NR2=NR2,
             NRKO=NRKO, rank_source='P32 single-head lepton-key knock-out (Holm-significant, ordered by effect)', ranks=RANK)
    pe, pr = [], []
    for c in CL:
        J, rows = P.load_rows(P.TEST / f'{PRE[c]}_{FILE}.root', NE + NREF, rng)
        perm = rng.permutation(len(rows)); pe.append(sub(J, np.sort(perm[:NE]))); pr.append(sub(J, np.sort(perm[NE:])))
        log(f'loaded {c}: {len(rows)} random rows of {PRE[c]}_{FILE}.root (eval {NE}, reference {NREF})')
    B = P.rebuild(cat(pe)); BREF = P.rebuild(cat(pr)); y = B['label'].astype(int); N = len(y); tok = P.lepton_idx(B)
    sub_idx = np.sort(np.concatenate([rng.choice(np.where(y == k)[0], NSUB, replace=False) for k in range(10)])); BS_ = sub(B, sub_idx)
    ys, toks = y[sub_idx], tok[sub_idx]
    lv = np.where((y == HQQL) & (tok >= 0))[0]; mh, mq = P.matched_hadron(sub(B, lv), tok[lv], rng)
    keep = mh >= 0; lv, mh, mq = lv[keep], mh[keep], mq[keep]; tl = tok[lv]; BL = sub(B, lv); r_ = np.arange(len(lv))
    ise = BL['isElectron'][r_, tl] > 0.5; mn = matched_neutral(BL, tl); okn = np.where(mn >= 0)[0]
    R['sample'] = dict(N=N, per_class=np.bincount(y, minlength=10).tolist(), ref_N=int(len(BREF['label'])), sub_N=int(len(sub_idx)),
                       no_lepton_frac=[float((tok[y == k] < 0).mean()) for k in range(10)], n_lv=int(len(lv)), electron_frac=float(ise.mean()),
                       lepton_rank0=float((tl == 0).mean()), match_abs_log_pt_ratio_median=float(np.median(mq)), n_noop=int(len(okn)))
    log('sample', R['sample'])
    E = dict(base=P.rebuild(BL), e1=P.ed_flag(BL, tl, 'isChargedHadron'), e2n=P.ed_flag(BL, mh, 'isNeutralHadron'), e1x=flavswap(BL, tl, ise))
    E0 = dict(base=P.rebuild(sub(BL, okn)), e0=noop(sub(BL, okn), mn[okn]))
    R['edit_checks'] = dict(e1_other_lepton_left=float((P.lepton_idx(E['e1']) >= 0).mean()),
                            e1_flag_cols_changed_only=bool(all(np.array_equal(E['e1'][k], E['base'][k]) for k in ('px', 'py', 'pz', 'energy', 'charge', 'd0val', 'dzval'))),
                            e0_identical=bool(all(np.array_equal(E0['e0'][k], E0['base'][k]) for k in E0['base'] if np.ndim(E0['base'][k]))))
    log('edit checks', R['edit_checks'])
    P1s = P.rebuild(sample(200 if SMOKE else 1000, 3000)); yP1 = P1s['label'].astype(int)
    Wall = P.boot_W(N, NB, rng); Wl = P.boot_W(len(lv), NB, rng); W0 = P.boot_W(len(okn), NB, rng)
    MODELS = [('ParT_full', 'part', 'full'), ('ParT_kinpid', 'part', 'kinpid'), ('ParT_kin', 'part', 'kin'),
              ('PN_full', 'pn', 'full'), ('PN_kinpid', 'pn', 'kinpid')]
    R['models'] = {}
    for name, kind, feat in MODELS:
        log(f'==== {name}'); res = {}; t0 = time.time()
        M = Run(name, feat) if kind == 'part' else P.PNRun(feat)
        res['acc_P1'] = P.acc_block(M.run(P1s), yP1)['acc10']
        L0 = M.run(B); res['sample'] = metr(L0, y, tok, Wall)
        log(f'  acc P1 {res["acc_P1"]:.4f}  sample acc10 {res["sample"]["acc10"]:.4f} {np.round(res["sample"]["acc10_ci"], 4).tolist()} '
            f'balanced {res["sample"]["balanced"]:.4f} Hqql recall {res["sample"]["hqql_recall"]:.4f} Tbl->Hqql {res["sample"]["tbl_called_hqql"]:.4f}')
        LE = {k: M.run(v) for k, v in E.items()}; ed = {}
        for k in ('e1', 'e2n', 'e1x'): ed[k] = P.edit_stats(LE['base'], LE[k], Wl)
        LE0 = {k: M.run(v) for k, v in E0.items()}; ed['e0'] = P.edit_stats(LE0['base'], LE0['e0'], W0)
        ed['e0']['max_abs_logit_diff'] = float(np.abs(LE0['e0'] - LE0['base']).max())
        for a, b in (('e1', 'e2n'), ('e1', 'e1x')):
            ed[f'{a}_vs_{b}'] = dict(mcnemar_changed=P.mcnemar(ed[b]['_ch'], ed[a]['_ch']), ddlogp=P.mstat(ed[a]['_d'] - ed[b]['_d'], Wl))
        ed['e1']['by_flavour'] = {f: dict(n=int(m.sum()), changed=float(ed['e1']['_ch'][m].mean()), dlogp=float(ed['e1']['_d'][m].mean()))
                                  for f, m in (('electron', ise), ('muon', ~ise))}
        ed['e1x']['by_flavour'] = {f: dict(n=int(m.sum()), changed=float(ed['e1x']['_ch'][m].mean()), dlogp=float(ed['e1x']['_d'][m].mean()))
                                   for f, m in (('electron_to_muon', ise), ('muon_to_electron', ~ise))}
        for k in ('e1', 'e2n', 'e1x', 'e0'):
            log(f'  {k:4s} changed {ed[k]["calls_changed"]["v"]:.4f} {np.round(ed[k]["calls_changed"]["ci"], 4).tolist()} dlogp {ed[k]["dlogp"]["v"]:+.3f} '
                f'{np.round(ed[k]["dlogp"]["ci"], 3).tolist()} Hqql {ed[k]["mcnemar_hqql"]["before"]:.3f}->{ed[k]["mcnemar_hqql"]["after"]:.3f} to {ed[k]["changed_to"]}')
        log(f'  e1 vs e2n McNemar {ed["e1_vs_e2n"]["mcnemar_changed"]}; e1 vs e1x {ed["e1_vs_e1x"]["mcnemar_changed"]}; e0 max |dlogit| {ed["e0"]["max_abs_logit_diff"]:.2e}')
        res['edits'] = ed
        if kind == 'part':
            res['knockout'] = knockouts(M, name, E['base'], LE['base'], tl, mh, Wl, rng)
            if name in RANK:
                res['circuit'] = circuits(M, name, B, y, tok, Wall, BS_, ys, toks, BREF, E, tl, Wl, rng)
            res['head_recon_max_abs'] = M.recon
        res['runtime_s'] = time.time() - t0
        R['models'][name] = res
        FN.write_text(json.dumps(P.strip(R), indent=1)); log(f'  wrote {FN} ({res["runtime_s"]:.0f} s)')
        del M; torch.cuda.empty_cache()
    R['runtime_s'] = time.time() - P.T0
    FN.write_text(json.dumps(P.strip(R), indent=1)); log(f'done; wrote {FN}')


def knockouts(M, name, BK, L0, tl, mh, Wl, rng):
    dom = DOM[name]; lo0 = P.lodds(L0); c0 = L0.argmax(1); out = {'dominant': P.hlab(dom)}

    def run(kind, heads, toks):
        f = M.kmask if kind == 'key' else M.vzero
        return M.run(BK, lambda idx, Pn: {'kmask' if kind == 'key' else 'vzero': f(len(idx), Pn, heads, toks[idx])})

    def st(La, Lb=None):
        a = P.lodds(La) - lo0; cha = La.argmax(1) != c0; e = dict(dlodds=P.mstat(a, Wl), changed=P.mstat(cha.astype(float), Wl),
                                                                    hqql_after=float((La.argmax(1) == HQQL).mean()))
        if Lb is not None:
            chb = Lb.argmax(1) != c0
            e.update(matched_dlodds=P.mstat(P.lodds(Lb) - lo0, Wl), matched_changed=P.mstat(chb.astype(float), Wl), mcnemar=P.mcnemar(chb, cha))
        return e
    cls16 = [x for x in HEADS if x[0] >= 8]; par64 = [x for x in HEADS if x[0] < 8]
    for kind in ('key', 'value'):
        for lab, hs in (('dominant', [dom]), ('class16', cls16), ('particle64', par64), ('all80', HEADS)):
            if kind == 'value' and lab == 'particle64': continue
            e = st(run(kind, hs, tl), run(kind, hs, mh)); out[f'{kind}_{lab}'] = e
            log(f'  KO[{kind}] {lab}: dlodds {e["dlodds"]["v"]:+.2f} {np.round(e["dlodds"]["ci"], 2).tolist()} changed {e["changed"]["v"]:.3f} '
                f'{np.round(e["changed"]["ci"], 3).tolist()} | matched {e["matched_dlodds"]["v"]:+.2f} changed {e["matched_changed"]["v"]:.3f}')
    others = [x for x in HEADS if x != dom]; rk = {}
    for k in KS_KO:
        sets = []
        for _ in range(NRKO):
            Rs = [others[j] for j in rng.choice(len(others), k, replace=False)]
            La, Lb = run('key', Rs, tl), run('key', Rs + [dom], tl)
            cha, chb = La.argmax(1) != c0, Lb.argmax(1) != c0
            sets.append(dict(heads=[P.hlab(x) for x in Rs], changed_R=float(cha.mean()), changed_domR=float(chb.mean()),
                             dlodds_R=float((P.lodds(La) - lo0).mean()), dlodds_domR=float((P.lodds(Lb) - lo0).mean()), mcnemar=P.mcnemar(cha, chb)))
        inc = np.array([s['changed_domR'] - s['changed_R'] for s in sets]); pv = np.array([s['mcnemar']['p'] for s in sets])
        rk[k] = dict(sets=sets, changed_R_mean=float(np.mean([s['changed_R'] for s in sets])), changed_domR_mean=float(np.mean([s['changed_domR'] for s in sets])),
                     increment_mean=float(inc.mean()), increment_range=[float(inc.min()), float(inc.max())],
                     increment_ci_over_sets=P.ci([rng.choice(inc, len(inc)).mean() for _ in range(1000)]),
                     n_sets_holm_sig=int((np.asarray(P.holm(pv)) < 0.05).sum()))
        log(f'  KO b{dom} + {k} random vs {k} random ({NRKO} sets): changed {rk[k]["changed_R_mean"]:.3f} -> {rk[k]["changed_domR_mean"]:.3f} '
            f'(increment {rk[k]["increment_mean"]:+.3f}, range {np.round(rk[k]["increment_range"], 3).tolist()}, Holm-sig sets {rk[k]["n_sets_holm_sig"]})')
    out['dominant_plus_random'] = rk
    return out


def circuits(M, name, B, y, tok, Wall, BS_, ys, toks, BREF, E, tl, Wl, rng):
    from interp.part_core import sub
    out = {}
    M.run(BREF, lambda idx, Pn: dict(rec=True)); M.fix_means()
    out['ref_n'] = int(len(BREF['label'])); out['mlp_mean_norm'] = {str(i): float(M.MM[i].norm()) for i in range(10)}

    def ev(keep, J=B, yy=y, tt=tok, W=Wall, mlp=True, conf=True):
        L = M.run(J, lambda idx, Pn: M.circ_ctl(set(keep), mlp)); return metr(L, yy, tt, W, conf), L
    out['full_model'], _ = ev(HEADS, mlp=False)
    out['full_model_sub'], _ = ev(HEADS, BS_, ys, toks, None, mlp=False, conf=False)
    out['nothing_kept'], _ = ev([]); out['all80_mlp_ablated'], _ = ev(HEADS)
    out['all80_particle_mlps_ablated'], _ = ev(HEADS, mlp=PMLP); out['all80_class_mlps_ablated'], _ = ev(HEADS, mlp=CMLP)
    log(f'  all 80 heads: particle MLPs at mean -> acc10 {out["all80_particle_mlps_ablated"]["acc10"]:.3f} balanced {out["all80_particle_mlps_ablated"]["balanced"]:.3f}; '
        f'class MLPs at mean -> acc10 {out["all80_class_mlps_ablated"]["acc10"]:.3f} balanced {out["all80_class_mlps_ablated"]["balanced"]:.3f}')
    log(f'  [check] full via hooks acc10 {out["full_model"]["acc10"]:.4f}; nothing kept: distinct calls {out["nothing_kept"]["n_distinct_calls"]} '
        f'pred share {np.round(out["nothing_kept"]["pred_share"], 3).tolist()}; all 80 heads, MLPs at mean: acc10 {out["all80_mlp_ablated"]["acc10"]:.3f} '
        f'balanced {out["all80_mlp_ablated"]["balanced"]:.3f} Hqql recall {out["all80_mlp_ablated"]["hqql_recall"]:.3f}')
    rk = {g: [lab2ih(s) for s in RANK[name][g]] for g in ('all', 'cls', 'par')}
    cls16 = [x for x in HEADS if x[0] >= 8]; par64 = [x for x in HEADS if x[0] < 8]
    ED = dict(base=E['base'], e1=E['e1'], e2n=E['e2n'])
    sel = {}
    for k in KS:
        for g in ('all', 'cls', 'par'):
            if len(rk[g]) < k: continue
            hs = rk[g][:k]; lab = f'{g}_k{k}'; e, _ = ev(hs); e['heads'] = [P.hlab(x) for x in hs]
            e['sub'], _ = ev(hs, BS_, ys, toks, None, conf=False)
            e['mlps_kept'], _ = ev(hs, mlp=False, conf=False)
            e['class_mlps_kept'], _ = ev(hs, mlp=PMLP, conf=False)       # only the 8 particle-block MLPs at their mean
            e['particle_mlps_kept'], _ = ev(hs, mlp=CMLP, conf=False)    # only the 2 class-block MLPs at their mean
            LF = {kk: M.run(v, lambda idx, Pn, hs=hs: M.circ_ctl(set(hs), True)) for kk, v in ED.items()}
            e['flag_edits'] = {kk: P.strip(P.edit_stats(LF['base'], LF[kk], Wl)) for kk in ('e1', 'e2n')}
            e['flag_edits']['hqql_recall_base'] = float((LF['base'].argmax(1) == HQQL).mean())
            e['flag_edits']['e1_vs_e2n'] = P.mcnemar(LF['e2n'].argmax(1) != LF['base'].argmax(1), LF['e1'].argmax(1) != LF['base'].argmax(1))
            sel[lab] = e
            fe = e['flag_edits']
            log(f'  circuit {lab} {e["heads"]}: recall {e["hqql_recall"]:.3f} {np.round(e["hqql_recall_ci"], 3).tolist()} balanced {e["balanced"]:.3f} '
                f'{np.round(e["balanced_ci"], 3).tolist()} acc10 {e["acc10"]:.3f} Tbl->Hqql {e["tbl_called_hqql"]:.3f} AUC rest {e["auc_hqql_vs_rest"]:.3f} '
                f'vs Tbl {e["auc_hqql_vs_tbl"]:.3f} | class MLPs kept: recall {e["class_mlps_kept"]["hqql_recall"]:.3f} balanced {e["class_mlps_kept"]["balanced"]:.3f} | '
                f'particle MLPs kept: recall {e["particle_mlps_kept"]["hqql_recall"]:.3f} balanced {e["particle_mlps_kept"]["balanced"]:.3f} | MLPs kept: recall {e["mlps_kept"]["hqql_recall"]:.3f} balanced {e["mlps_kept"]["balanced"]:.3f} | '
                f'e1: Hqql {fe["e1"]["mcnemar_hqql"]["before"]:.3f}->{fe["e1"]["mcnemar_hqql"]["after"]:.3f}, e2n -> {fe["e2n"]["mcnemar_hqql"]["after"]:.3f}')
    out['selected'] = sel
    # ---- random controls on the subsample
    rnd = {}
    singles = []
    for h in HEADS:
        e, _ = ev([h], BS_, ys, toks, None, conf=False); e['heads'] = [P.hlab(h)]; singles.append(e)
    rnd['k1_exhaustive'] = dict(sets=singles, summary=summ(singles))
    for k in KS[1:]:
        hs = rk['all'][:k]; nc = sum(1 for x in hs if x[0] >= 8); rv = []
        for _ in range(NR):
            s = [cls16[j] for j in rng.choice(16, nc, replace=False)] + [par64[j] for j in rng.choice(64, k - nc, replace=False)]
            e, _ = ev(s, BS_, ys, toks, None, conf=False); e['heads'] = [P.hlab(x) for x in s]; rv.append(e)
        rnd[f'all_k{k}'] = dict(n=NR, n_class=nc, sets=rv, summary=summ(rv))
        for g, pool in (('cls', cls16), ('par', par64)):
            if f'{g}_k{k}' not in sel: continue
            rv2 = []
            for _ in range(NR2):
                s = [pool[j] for j in rng.choice(len(pool), k, replace=False)]
                e, _ = ev(s, BS_, ys, toks, None, conf=False); e['heads'] = [P.hlab(x) for x in s]; rv2.append(e)
            rnd[f'{g}_k{k}'] = dict(n=NR2, sets=rv2, summary=summ(rv2))
    rnd['cls_k1'] = dict(sets=[s for s, h in zip(singles, HEADS) if h[0] >= 8]); rnd['cls_k1']['summary'] = summ(rnd['cls_k1']['sets'])
    rnd['par_k1'] = dict(sets=[s for s, h in zip(singles, HEADS) if h[0] < 8]); rnd['par_k1']['summary'] = summ(rnd['par_k1']['sets'])
    rnd['all_k1'] = rnd['k1_exhaustive']
    for lab, e in sel.items():
        g, k = lab.split('_k'); key = f'{g}_k{k}'
        if key not in rnd: continue
        rs = rnd[key]['sets']; s = e['sub']
        e['vs_random'] = {m: dict(selected_sub=s[m], p=emp_p(s[m], [r[m] for r in rs]), random=rnd[key]['summary'][m])
                          for m in ('hqql_recall', 'balanced', 'acc10', 'auc_hqql_vs_rest', 'auc_hqql_vs_tbl')}
        e['vs_random']['n_random'] = len(rs)
        v = e['vs_random']
        log(f'  {lab} vs {len(rs)} random: recall {s["hqql_recall"]:.3f} (random mean {v["hqql_recall"]["random"]["mean"]:.3f}, q95 {v["hqql_recall"]["random"]["q95"]:.3f}, '
            f'p {v["hqql_recall"]["p"]:.3f}); balanced {s["balanced"]:.3f} (mean {v["balanced"]["random"]["mean"]:.3f}, q95 {v["balanced"]["random"]["q95"]:.3f}, p {v["balanced"]["p"]:.3f}); '
            f'AUC vs Tbl {s["auc_hqql_vs_tbl"]:.3f} (mean {v["auc_hqql_vs_tbl"]["random"]["mean"]:.3f}, p {v["auc_hqql_vs_tbl"]["p"]:.3f})')
    for kk in rnd:
        for r in rnd[kk]['sets']: r.pop('recall', None)
    out['random'] = rnd
    return out


# ------------------------------------------------------------------------------------------------ figure
def plot():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    R = json.loads(FN.read_text()); Cf = R['models']['ParT_full']['circuit']; sel, rnd = Cf['selected'], Cf['random']
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.9), gridspec_kw=dict(width_ratios=[1, 1.1]))
    xs = np.arange(len(KS)); blue, grey = '#1f6fb4', '0.55'
    for i, k in enumerate(KS):
        rs = [r['auc_hqql_vs_rest'] for r in rnd[f'all_k{k}']['sets']]
        ax[0].scatter(i + np.random.default_rng(i).uniform(-0.12, 0.12, len(rs)), rs, s=8, color=grey, alpha=0.45, lw=0)
    ax[0].plot(xs, [sel[f'all_k{k}']['sub']['auc_hqql_vs_rest'] for k in KS], 'D', color=blue, ms=8, label='selected heads (top-k lepton heads)')
    ax[0].scatter([], [], s=14, color=grey, label='random head sets, same size and block mix')
    ax[0].axhline(Cf['full_model_sub']['auc_hqql_vs_rest'], color='k', ls=':', lw=1); ax[0].text(-0.3, Cf['full_model_sub']['auc_hqql_vs_rest'] - 0.035, 'full network', fontsize=8)
    ax[0].axhline(0.5, color='0.8', lw=0.8)
    ax[0].set_xticks(xs); ax[0].set_xticklabels([f'k = {k}' for k in KS]); ax[0].set_ylim(0.2, 1.02)
    ax[0].set_xlabel('heads kept (all other heads and all 10 MLPs at their mean)')
    ax[0].set_ylabel("AUC of the H→ℓνqq′ score, H→ℓνqq′ vs other 9 classes")
    ax[0].set_title('Literal circuit: the H→ℓνqq′ score separates the class,\nbut the call is the same for every jet (recall 0)', fontsize=10)
    ax[0].legend(fontsize=8, loc='lower right')
    e = sel['all_k3']; var = [('all 10 MLPs\nat mean', e), ('only the 8 particle-\nblock MLPs kept', e['particle_mlps_kept']),
                              ('only the 2 class-\nblock MLPs kept', e['class_mlps_kept']), ('all 10 MLPs\nkept', e['mlps_kept']), ('full\nnetwork', Cf['full_model'])]
    w = 0.38; xv = np.arange(len(var))
    b1 = ax[1].bar(xv - w / 2, [v['hqql_recall'] for _, v in var], w, color=blue, label='H→ℓνqq′ jets called H→ℓνqq′ (recall)')
    b2 = ax[1].bar(xv + w / 2, [v['tbl_called_hqql'] for _, v in var], w, color='#d1495b', label='t→bℓν jets called H→ℓνqq′')
    for bb in (b1, b2):
        for r in bb: ax[1].text(r.get_x() + r.get_width() / 2, r.get_height() + 0.01, f'{r.get_height():.2f}', ha='center', fontsize=7.5)
    ax[1].set_xticks(xv); ax[1].set_xticklabels([n for n, _ in var], fontsize=8); ax[1].set_ylim(0, 1.12); ax[1].set_ylabel('fraction of jets')
    ax[1].set_title('Top-3 lepton heads (b2h1, p7h8, p8h6) kept: which MLPs\nturn the score into an H→ℓνqq′ call (1,500 jets per class)', fontsize=10)
    ax[1].legend(fontsize=8, loc='upper left')
    fig.suptitle('ParT_full, JetClass test jets (_116, seed 3416); other heads replaced by their mean output', fontsize=10, y=1.0)
    fig.tight_layout(); fo = P.OUTD / 'p34_lepton_circuit.png'; fig.savefig(fo, dpi=140, bbox_inches='tight'); log(f'wrote {fo}')


if __name__ == '__main__':
    plot() if P.ARGS.plot else main()
