"""H35 (loop D, new D claim D14, b-prong localisation): do JetClass full-input taggers find the displaced tracks on the
prong that is the b of t -> bW, or only count displaced tracks anywhere in the jet (G194: all displaced tracks made prompt drops
ParT_full's top call 96 -> 68 %)?

Sample (random rows, never first-k)
  JetClass test TTBar_100 (Tbqq) and ZJetsToNuNu_100 (QCD), stored 550 < jet_pt < 650 GeV, |jet_eta| < 2, >= 3 particles,
  2,000 per class (h19_wpair_cross.jc_read, rng 3501 / 3502). Every variant (identity included) is rebuilt with
  part_core.build (eta = jet_eta + sign(jet_eta) deta); the kinematics are never changed, only (d0val, d0err, dzval, dzerr).
Track displacement: charged particles; S = |d0val| / d0err (G194); displaced = S > 2, prompt = charged with S <= 2.
Prong roles: three exclusive-kT subjets (E-scheme, h4_top_wmass.kt3_batch); W pair (a, b) = the pair with mass closest to
  80.4 GeV; c = the third subjet ("b-prong"). Cross-check: assignment agrees if n_disp(c) > max(n_disp(a), n_disp(b)).
Edits on top jets (all keep every particle's four-momentum; tuples t = (d0val, d0err, dzval, dzerr) are exchanged or zeroed)
  M    target W prong w = the one of (a, b) with more prompt tracks; the k = min(n_disp(c), n_prompt(w)) highest-pT displaced
       tracks of c exchange t with the k highest-pT prompt tracks of w, rank i with rank i. The jet's displaced-track count and
       the multiset of S values are unchanged; n_disp(c) -> n_disp(c) - k, n_disp(w) -> n_disp(w) + k.
  C1   the same exchange inside c: the k1 = min(k, n_prompt(c)) highest-pT displaced tracks of c with the k1 highest-pT prompt
       tracks of c (per-prong counts unchanged; displacement moves onto other tracks of the b-prong).
  C2   inside w: the k highest-pT prompt tracks of w (M's targets) exchange t with the next k2 = min(k, rest) charged tracks of w.
  R    every displaced track of c made prompt (d0val, dzval x 0; errors kept; G194 convention).
  Rk / RWk   kk = min(n_disp(c), n_disp(a) + n_disp(b)): the kk highest-pT displaced tracks of c (Rk), or of a and b (RWk), made
       prompt (equal number of tracks; per-track comparison).
  ALL  every displaced track of the jet made prompt (G194 anchor).
QCD jets: the analogous edits between the two highest-pT subjets: source = the one with more displaced tracks (tie: the harder),
  target = the other; M, C1, C2, R, ALL as above.
Models: ParT_full, PN_full (17 inputs incl. d0/dz), ParT_kin, PN_kin (7 kinematic inputs: zero by construction), ParT_perm =
  ParT_full with every weight tensor's entries permuted (torch seed 5, P13 convention).
Score: top log-odds lt = log p(Tbqq) / (1 - p(Tbqq)) of the 10-class softmax; top call = argmax == Tbqq.
References (sklearn, trained on JetClass training rows: train_100M TTBar_000 / ZJetsToNuNu_000, same window, NREF per class,
  rng 3503 / 3504), 5 retrains each (80 % row subsample per seed; MLP random_state = seed):
  Jref  (m, tau32, m_Wpair, n_disp_jet)         jet-level displaced-track count
  Sref  Jref + (n_disp(c), max / min n_disp(a, b))   classical subjet b-tag count
  LR (logistic regression) and MLP(64, 64) for each; response = mean over the 5 retrains of the change in log-odds.
Statistics per model x set x contrast: mean, 95 % bootstrap CI over jets (B = 2000), two-sided sign-flip permutation p (H0: mean
  = 0), Holm over all tested (model, set, contrast) cells; McNemar M vs C1 on top calls; ratio to Sref-MLP with bootstrap CI.
Run: ./sync.sh && ./part_job.sh d-h35-run 'python -m interp.h35_top_bprong'   (H35_SMOKE=1: 200 per class, 2,000 ref jets)
Outputs: $AI_RESULTS/physics/h35_top_bprong.json, h35_perjet.npz, h35_bprong.png
"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import h4_top_wmass as H          # noqa: E402
from interp import h19_wpair_cross as X       # noqa: E402

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
SMOKE = bool(os.environ.get('H35_SMOKE'))
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
TAG = 'h35_smoke' if SMOKE else 'h35'
PER = 200 if SMOKE else 2000
NREF = 2000 if SMOKE else 20000
B = 300 if SMOKE else 2000
NPERM = 500 if SMOKE else 5000
TBQQ, QCD = 8, 0
IPK = ('d0val', 'd0err', 'dzval', 'dzerr')
VARS = ('M', 'C1', 'C2', 'R', 'Rk', 'RWk', 'ALL')
NETS = ('ParT_full', 'PN_full', 'ParT_kin', 'PN_kin', 'ParT_perm')
TRAIN = Path('/j-jepa-vol/JetClass/Pythia/train_100M')


# ================================================================================================================ data
def read_window(path, per, seed):
    """per random rows of a JetClass file among 550 < jet_pt < 650, |jet_eta| < 2, >= 3 particles (as h19 jc_read)"""
    import uproot
    from interp.part_core import read_root, cat, sub
    t = uproot.open(path)['tree']; a = t.arrays(['jet_pt', 'jet_eta', 'jet_nparticles'], library='np')
    ok = (a['jet_pt'] > 550) & (a['jet_pt'] < 650) & (np.abs(a['jet_eta']) < 2) & (a['jet_nparticles'] >= 3)
    el = np.flatnonzero(ok); n = min(per, len(el)); idx = np.sort(np.random.default_rng(seed).choice(el, n, replace=False))
    parts = []; N = t.num_entries
    for s in range(0, N, 20000):
        loc = idx[(idx >= s) & (idx < s + 20000)] - s
        if len(loc): parts.append(sub(read_root(path, start=s, stop=min(s + 20000, N)), loc))
    log(f'{Path(path).name}: {len(el)} eligible of {N}; drew {n} (rng {seed})')
    return cat(parts), dict(file=Path(path).name, eligible=int(len(el)), rows=int(N), drawn=int(n))


def sig(J):
    ch = (J['mask'] > 0) & (J['charge'] != 0)
    return ch, np.where(ch, np.abs(J['d0val']) / np.maximum(J['d0err'], 1e-6), 0.0)


def roles(P4, nn, LAB):
    """W pair (a, b), b-prong c, W-pair mass, subjet pT (J, 3)"""
    J, L = LAB.shape; real = np.arange(L)[None, :] < nn[:, None]
    oh = np.stack([(LAB == k) & real for k in range(3)], -1).astype(np.float64)
    S = np.einsum('jlk,jld->jkd', oh, P4[:, :L].astype(np.float64))
    pairs = [(0, 1), (0, 2), (1, 2)]
    pm = np.stack([np.sqrt(np.maximum(H_m2(S[:, i] + S[:, j]), 0)) for i, j in pairs], 1)
    w = np.argmin(np.abs(pm - H.MWREF), 1); ab = np.array(pairs)[w]; a, b = ab[:, 0], ab[:, 1]; c = 3 - a - b
    return dict(a=a, b=b, c=c, mW=pm[np.arange(J), w], spt=np.hypot(S[..., 0], S[..., 1]))


def H_m2(v):
    return v[..., 3] ** 2 - (v[..., :3] ** 2).sum(-1)


def counts(J, LAB):
    """n_disp per subjet (J, 3) and n_prompt per subjet"""
    ch, S = sig(J); L = LAB.shape[1]; d = (ch & (S > 2))[:, :L]; p = (ch & (S <= 2))[:, :L]
    return np.stack([(d & (LAB == k)).sum(1) for k in range(3)], 1), np.stack([(p & (LAB == k)).sum(1) for k in range(3)], 1)


def ref_feats(O, R, nd):
    r = np.arange(len(nd)); ndc = nd[r, R['c']]; na, nb = nd[r, R['a']], nd[r, R['b']]
    jref = np.column_stack([O['m'], O['tau32'], R['mW'], nd.sum(1)])
    return dict(Jref=jref, Sref=np.column_stack([jref, ndc, np.maximum(na, nb), np.minimum(na, nb)]))


# ================================================================================================================ edits
def make_edits(J, LAB, R, y):
    """edited copies of the impact-parameter arrays for every variant; per-jet bookkeeping"""
    N = len(y); ch, S = sig(J); pt = np.hypot(J['px'], J['py']); L = LAB.shape[1]
    E = {v: {k: J[k].copy() for k in IPK} for v in VARS}
    info = {k: np.zeros(N, np.int64) for k in ('k', 'k1', 'k2', 'kk', 'nR', 'nALL', 'src', 'tgt')}
    nd, npr = counts(J, LAB)

    def swap(v, j, i1, i2):
        for k in IPK:
            A = E[v][k]; t = A[j, i1].copy(); A[j, i1] = A[j, i2]; A[j, i2] = t

    def prompt(v, j, i):
        E[v]['d0val'][j, i] = 0.0; E[v]['dzval'][j, i] = 0.0

    for j in range(N):
        lab = LAB[j]; n = int((lab >= 0).sum())
        disp = np.flatnonzero(ch[j, :n] & (S[j, :n] > 2)); prm = np.flatnonzero(ch[j, :n] & (S[j, :n] <= 2)); chg = np.flatnonzero(ch[j, :n])
        byp = lambda idx: idx[np.argsort(-pt[j, idx], kind='stable')]
        if y[j] == 1:
            src = R['c'][j]; a, b = R['a'][j], R['b'][j]
            tgt = a if (npr[j, a] > npr[j, b] or (npr[j, a] == npr[j, b] and nd[j, a] <= nd[j, b])) else b
        else:
            o = np.argsort(-R['spt'][j]); s1, s2 = o[0], o[1]
            src, tgt = (s1, s2) if nd[j, s1] >= nd[j, s2] else (s2, s1)
        info['src'][j], info['tgt'][j] = src, tgt
        Dc = byp(disp[lab[disp] == src]); Pc = byp(prm[lab[prm] == src]); Pw = byp(prm[lab[prm] == tgt])
        k = min(len(Dc), len(Pw)); info['k'][j] = k
        if k: swap('M', j, Dc[:k], Pw[:k])
        k1 = min(k, len(Pc)); info['k1'][j] = k1
        if k1: swap('C1', j, Dc[:k1], Pc[:k1])
        rest = byp(np.setdiff1d(chg[lab[chg] == tgt], Pw[:k])); k2 = min(k, len(rest)); info['k2'][j] = k2
        if k2: swap('C2', j, Pw[:k2], rest[:k2])
        if len(Dc): prompt('R', j, Dc); info['nR'][j] = len(Dc)
        if len(disp): prompt('ALL', j, disp); info['nALL'][j] = len(disp)
        if y[j] == 1:
            Dw = byp(disp[(lab[disp] == R['a'][j]) | (lab[disp] == R['b'][j])]); kk = min(len(Dc), len(Dw)); info['kk'][j] = kk
            if kk: prompt('Rk', j, Dc[:kk]); prompt('RWk', j, Dw[:kk])
    return E, info


def variant_dict(J, Ev):
    from interp.part_core import build, kin
    ex = dict(J); ex.update(Ev); return build(J, *kin(J), extra=ex)


# ================================================================================================================ models
class Nets:
    def __init__(self):
        import torch
        from interp.part_core import ParTNetwork
        from interp.pn_net import load_particlenet
        from jetdistill.config import MODELS
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'; self.m = {}
        for n in ('full', 'kin'): self.m[f'ParT_{n}'] = ('part', n, ParTNetwork(n))
        for n, dims in (('full', 17), ('kin', 7)):
            self.m[f'PN_{n}'] = ('pn', n, load_particlenet(Path(MODELS).parent / 'particlenet' / f'ParticleNet_{n}.pt', dims, dev))
        netp = ParTNetwork('full'); g = torch.Generator().manual_seed(5)
        with torch.no_grad():
            for p in netp.model.parameters():
                if p.numel() > 1: p.copy_(p.flatten()[torch.randperm(p.numel(), generator=g).to(p.device)].view_as(p))
        self.m['ParT_perm'] = ('part', 'full', netp); log(f'models loaded on {dev}')

    def logits(self, name, Jd):
        from interp.part_core import inputs
        from interp.p1v_verify_part import fwd
        from interp.h1_irc_panel import pn_logits
        kind, n, net = self.m[name]
        return fwd(net, *inputs(Jd, n))[1] if kind == 'part' else pn_logits(net, Jd, n)


def top_lo(L):
    L = np.asarray(L, np.float64); o = np.delete(L, TBQQ, 1); mx = o.max(1, keepdims=True)
    return L[:, TBQQ] - (mx[:, 0] + np.log(np.exp(o - mx).sum(1)))


lo = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))


def fit_refs(F, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    R, info = {}, {}
    for fs in ('Jref', 'Sref'):
        for kind in ('LR', 'MLP'):
            ms, accs = [], []
            for s in range(5):
                rr = np.random.default_rng(3600 + s).random(len(y)) < 0.8; Xt = F[fs][rr]; sc = StandardScaler().fit(Xt)
                m = (LogisticRegression(max_iter=2000) if kind == 'LR' else
                     MLPClassifier(hidden_layer_sizes=(64, 64), early_stopping=True, random_state=s, max_iter=300)).fit(sc.transform(Xt), y[rr])
                accs.append(float(m.score(sc.transform(F[fs][~rr]), y[~rr]))); ms.append((m, sc))
            R[f'{fs}_{kind}'] = (fs, ms); info[f'{fs}_{kind}'] = dict(heldout_acc=accs)
            log(f'ref {fs}_{kind}: held-out acc {np.mean(accs):.4f} (5 retrains, {np.min(accs):.4f}-{np.max(accs):.4f})')
    return R, info


def eval_refs(R, F):
    """{name: (5, J) log-odds}"""
    return {k: np.stack([lo(m.predict_proba(sc.transform(F[fs]))[:, 1]) for m, sc in ms]) for k, (fs, ms) in R.items()}


# ================================================================================================================ stats
def boot(x, rng, b=B):
    x = np.asarray(x, float)
    if len(x) == 0: return float('nan'), [float('nan')] * 2
    m = x[rng.integers(0, len(x), (b, len(x)))].mean(1); return float(x.mean()), np.percentile(m, [2.5, 97.5]).tolist()


def perm2(x, rng, n=NPERM):
    x = np.asarray(x, float)
    if len(x) == 0 or np.all(x == 0): return 1.0
    obs = abs(x.mean()); s = rng.choice([-1.0, 1.0], (n, len(x))); return float((np.sum(np.abs((s * x).mean(1)) >= obs - 1e-15) + 1) / (n + 1))


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); adj = np.empty_like(ps); run = 0.0
    for r, i in enumerate(o): run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj


def ratio_ci(a, b, rng, nb=B):
    if len(a) == 0 or abs(b.mean()) < 1e-12: return float('nan'), [float('nan')] * 2
    i = rng.integers(0, len(a), (nb, len(a))); r = a[i].mean(1) / b[i].mean(1)
    return float(a.mean() / b.mean()), np.percentile(r, [2.5, 97.5]).tolist()


# ================================================================================================================ main
def main():
    from interp.part_core import cat
    RES.mkdir(parents=True, exist_ok=True); out = dict(node='H35', smoke=SMOKE, per_class=PER, n_ref=NREF, B=B, n_perm=NPERM)
    # ---- sample
    Jt, it = X.jc_read('Tbqq', 100, PER, 3501)[::2]; Jq, iq = X.jc_read('QCD', 100, PER, 3502)[::2]
    J = cat([Jt, Jq]); y = (J['label'] == TBQQ).astype(int); out['sample'] = dict(Tbqq=it, QCD=iq, n_top=int(y.sum()), n=int(len(y)))
    P4, nn = X.jc_P4(J); LAB = H.kt3_batch(np.ascontiguousarray(P4.astype(np.float64)), nn); R = roles(P4, nn, LAB)
    O = H._obs(P4, nn); nd0, np0 = counts(J, LAB); r = np.arange(len(y)); log(f'sample {len(y)} jets; kT-3 roles and observables done')
    ndc = nd0[r, R['c']]; ndw = nd0[r, R['a']] + nd0[r, R['b']]; agree = ndc > np.maximum(nd0[r, R['a']], nd0[r, R['b']])
    t = y == 1
    out['roles'] = dict(top_ndisp_c_mean=float(ndc[t].mean()), top_ndisp_W_mean=float(ndw[t].mean()), top_ndisp_jet_mean=float(nd0[t].sum(1).mean()),
                        top_agree_frac=float(agree[t].mean()), top_agree_or_tie_frac=float((ndc >= np.maximum(nd0[r, R['a']], nd0[r, R['b']]))[t].mean()),
                        top_c_has_disp_frac=float((ndc[t] > 0).mean()), qcd_ndisp_jet_mean=float(nd0[~t].sum(1).mean()),
                        top_mW_median=float(np.median(R['mW'][t])), top_m_median=float(np.median(O['m'][t])))
    log('roles', out['roles'])
    # ---- edits
    E, info = make_edits(J, LAB, R, y)
    for v in VARS:
        Jv = dict(J); Jv.update(E[v]); ndv, _ = counts(Jv, LAB)
        if v in ('M', 'C1', 'C2'):
            assert np.array_equal(ndv.sum(1), nd0.sum(1)), f'{v} changed the jet displaced count'
            assert np.allclose(np.sort(sig(Jv)[1], 1), np.sort(sig(J)[1], 1)), f'{v} changed the S multiset'
        out.setdefault('edit_check', {})[v] = dict(top_ndisp_c_after=float(ndv[r, R['c']][t].mean()), top_ndisp_jet_after=float(ndv.sum(1)[t].mean()),
                                                     qcd_ndisp_jet_after=float(ndv.sum(1)[~t].mean()))
    out['edit_info'] = {k: dict(top_mean=float(v[t].mean()), qcd_mean=float(v[~t].mean())) for k, v in info.items() if k not in ('src', 'tgt')}
    log('edit bookkeeping', out['edit_info'])
    # ---- networks
    nets = Nets(); Jid = variant_dict(J, {k: J[k] for k in IPK}); Jvar = {v: variant_dict(J, E[v]) for v in VARS}
    LT = {}; CALL = {}; out['unedited'] = {}
    for name in NETS:
        Lid = nets.logits(name, Jid); LT[(name, 'id')] = top_lo(Lid); CALL[(name, 'id')] = Lid.argmax(1) == TBQQ
        lab10 = np.where(y == 1, TBQQ, QCD)
        out['unedited'][name] = dict(acc_10class=float((Lid.argmax(1) == lab10).mean()), acc_binary=float(((Lid[:, TBQQ] - Lid[:, QCD] > 0) == (y == 1)).mean()),
                                     AUC=float(H._auc(Lid[:, TBQQ] - Lid[:, QCD], y)), top_call_rate_tops=float(CALL[(name, 'id')][t].mean()))
        for v in VARS:
            L = nets.logits(name, Jvar[v]); LT[(name, v)] = top_lo(L); CALL[(name, v)] = L.argmax(1) == TBQQ
        log(f'{name}: ' + json.dumps(out['unedited'][name]))
    Lraw = nets.logits('ParT_full', J)
    out['input_path_check'] = dict(max_abs_toplo_raw_vs_build=float(np.abs(top_lo(Lraw) - LT[('ParT_full', 'id')]).max()))
    out['kin_sanity'] = {n: {v: float(np.abs(LT[(n, v)] - LT[(n, 'id')]).max()) for v in VARS} for n in ('ParT_kin', 'PN_kin')}
    log('kin sanity (max |d|):', out['kin_sanity'], 'input path', out['input_path_check'])
    # ---- references
    Fr, yr = [], []
    for c, lab in (('TTBar', 1), ('ZJetsToNuNu', 0)):
        Jr, inf = read_window(TRAIN / f'{c}_000.root', NREF, 3503 + (1 - lab)); out.setdefault('ref_rows', {})[c] = inf
        P4r, nnr = X.jc_P4(Jr); LABr = H.kt3_batch(np.ascontiguousarray(P4r.astype(np.float64)), nnr)
        Fr.append(ref_feats(H._obs(P4r, nnr), roles(P4r, nnr, LABr), counts(Jr, LABr)[0])); yr.append(np.full(len(nnr), lab))
    Fr = {k: np.concatenate([f[k] for f in Fr]) for k in Fr[0]}; yr = np.concatenate(yr)
    REF, out['ref_info'] = fit_refs(Fr, yr)
    RL = {}
    for v in ('id',) + VARS:
        Jv = dict(J)
        if v != 'id': Jv.update(E[v])
        for k, s in eval_refs(REF, ref_feats(O, R, counts(Jv, LAB)[0])).items(): RL[(k, v)] = s
    REFN = list(REF)
    out['ref_unedited'] = {k: dict(acc_binary=float(((RL[(k, 'id')].mean(0) > 0) == (y == 1)).mean()), AUC=float(H._auc(RL[(k, 'id')].mean(0), y))) for k in REFN}
    log('refs unedited', out['ref_unedited'])
    # ---- statistics
    rng = np.random.default_rng(3599)
    sets = {'top_agree': t & agree & (info['k'] > 0), 'top_all': t & (info['k'] > 0), 'qcd': ~t & (info['k'] > 0),
            'top_kk': t & (info['kk'] > 0)}
    out['sets'] = {k: int(v.sum()) for k, v in sets.items()}; log('sets', out['sets'])

    def dlt(m, v):
        if m in NETS: return LT[(m, v)] - LT[(m, 'id')]
        return (RL[(m, v)] - RL[(m, 'id')]).mean(0)

    def dlt_retr(m, v):
        return RL[(m, v)] - RL[(m, 'id')]
    CON = {'M': lambda m: dlt(m, 'M'), 'C1': lambda m: dlt(m, 'C1'), 'C2': lambda m: dlt(m, 'C2'), 'M-C1': lambda m: dlt(m, 'M') - dlt(m, 'C1'),
           'M-C2': lambda m: dlt(m, 'M') - dlt(m, 'C2'), 'R': lambda m: dlt(m, 'R'), 'ALL': lambda m: dlt(m, 'ALL'),
           'Rk': lambda m: dlt(m, 'Rk'), 'RWk': lambda m: dlt(m, 'RWk'), 'Rk-RWk': lambda m: dlt(m, 'Rk') - dlt(m, 'RWk')}
    TESTED = {'M-C1', 'M-C2', 'Rk-RWk'}
    res, cells = {}, []
    for sn, sm in sets.items():
        for m in NETS + tuple(REFN):
            for cn, f in CON.items():
                if sn != 'top_kk' and cn in ('Rk', 'RWk', 'Rk-RWk'): continue
                if sn == 'top_kk' and cn not in ('Rk', 'RWk', 'Rk-RWk', 'R', 'ALL'): continue
                x = f(m)[sm]; mu, ci = boot(x, rng); p = perm2(x, rng)
                c = dict(mean=mu, ci=ci, p_perm=p, n=int(sm.sum()))
                if cn in ('Rk', 'RWk', 'Rk-RWk', 'R', 'ALL'):
                    den = info['kk'] if cn in ('Rk', 'RWk', 'Rk-RWk') else (info['nR'] if cn == 'R' else info['nALL'])
                    ok = sm & (den > 0); c['per_track_mean'], c['per_track_ci'] = boot(f(m)[ok] / den[ok], rng)
                if m in REFN:
                    v1, v2 = (cn.split('-') + [None])[:2]
                    rr = dlt_retr(m, v1)[:, sm] - (dlt_retr(m, v2)[:, sm] if v2 else 0)
                    c['retrain_means'] = [float(a) for a in rr.mean(1)]
                res[f'{sn}|{m}|{cn}'] = c
                if cn in TESTED and m in NETS: cells.append(f'{sn}|{m}|{cn}')
            if m in NETS and sn != 'top_kk':
                a, b = CALL[(m, 'C1')][sm], CALL[(m, 'M')][sm]
                for v in ('id',) + VARS: res[f'{sn}|{m}|topcall|{v}'] = float(CALL[(m, v)][sm].mean())
                res[f'{sn}|{m}|mcnemar_M_vs_C1'] = H._mcnemar(a, b)
                for rn in ('Sref_MLP', 'Sref_LR'):
                    rv, ci = ratio_ci(CON['M-C1'](m)[sm], CON['M-C1'](rn)[sm], rng)
                    res[f'{sn}|{m}|ratio_M-C1_to_{rn}'] = dict(ratio=rv, ci=ci)
                xa, xb = CON['M-C1'](m)[sm], CON['M-C1']('Sref_MLP')[sm]
                res[f'{sn}|{m}|corr_M-C1_with_Sref_MLP'] = float(np.corrcoef(xa, xb)[0, 1]) if xa.std() > 0 and xb.std() > 0 else float('nan')
    adj = holm([res[c]['p_perm'] for c in cells])
    for c, a in zip(cells, adj): res[c]['p_holm'] = float(a)
    out['holm_family'] = dict(n_tests=len(cells), cells=cells); out['results'] = res
    for c in cells:
        if c.startswith('top_agree') or c.startswith('qcd'):
            log(f'{c}: {res[c]["mean"]:+.3f} [{res[c]["ci"][0]:+.3f}, {res[c]["ci"][1]:+.3f}] p_holm {res[c]["p_holm"]:.2g}')
    for sn in ('top_agree', 'qcd'):
        for m in REFN: log(f'{sn}|{m}|M-C1: {res[f"{sn}|{m}|M-C1"]["mean"]:+.3f}')
    # ---- save
    def clean(o):
        if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [clean(v) for v in o]
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return clean(o.tolist())
        return o
    (RES / f'{TAG}_top_bprong.json').write_text(json.dumps(clean(out), indent=1))
    np.savez_compressed(RES / f'{TAG}_perjet.npz', y=y, agree=agree, ndc=ndc, ndw=ndw, **{f'k_{k}': v for k, v in info.items()},
                        **{f'lt|{m}|{v}': a for (m, v), a in LT.items()}, **{f'ref|{m}|{v}': a for (m, v), a in RL.items()})
    log('saved'); fig(out)


def fig(out):
    import matplotlib
    matplotlib.use('Agg'); import matplotlib.pyplot as plt
    R = out['results']; models = [('ParT_full', 'ParT_full'), ('PN_full', 'ParticleNet_full'), ('ParT_perm', 'ParT_full, weights permuted'),
                                  ('Sref_MLP', 'classifier with subjet b-count'), ('Jref_MLP', 'classifier with jet-level count')]
    cons = [('M', 'move b-prong\ndisplacement to W prong'), ('C1', 'swap within\nb-prong'), ('C2', 'swap within\nW prong'), ('R', 'b-prong tracks\nmade prompt'), ('ALL', 'all tracks\nmade prompt')]
    fig_, ax = plt.subplots(1, 1, figsize=(12, 4.6)); w = 0.16; cols = ['#2f5d8a', '#c0603a', '#9a9a9a', '#5a9a5a', '#b5a33a']
    for i, (m, lab) in enumerate(models):
        mu = [R[f'top_agree|{m}|{c}']['mean'] for c, _ in cons]; ci = np.array([R[f'top_agree|{m}|{c}']['ci'] for c, _ in cons])
        xs = np.arange(len(cons)) + (i - 2) * w
        ax.bar(xs, mu, w, color=cols[i], label=lab, yerr=[np.array(mu) - ci[:, 0], ci[:, 1] - np.array(mu)], capsize=2)
    ax.axhline(0, color='k', lw=0.6); ax.set_xticks(range(len(cons))); ax.set_xticklabels([c for _, c in cons], fontsize=9)
    ax.set_ylabel('change in top log-odds\n(log p(top)/(1-p(top)))')
    ax.set_title(f'JetClass top jets, 550-650 GeV, N = {out["sets"]["top_agree"]}: moving track displacement off the b-prong '
                 'lowers the top score more than swaps inside a prong', fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1.0)); fig_.tight_layout()
    fig_.savefig(RES / f'{TAG}_bprong.png', dpi=130); log('figure saved')


if __name__ == '__main__':
    if os.environ.get('H35_FIG'): fig(json.loads((RES / f'{TAG}_top_bprong.json').read_text()))
    else: main()
