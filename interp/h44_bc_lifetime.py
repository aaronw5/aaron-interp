"""H44 (loop D; D2 / D14 extension, exploratory): to separate H->bb from H->cc, do the JetClass ParT_full and ParticleNet_full
read the size of the track displacement (lifetime) and/or the mass of the displaced-track system (vertex-mass proxy), or only the
number of displaced tracks? D2 ('multiplicity, not lifetime') was measured on H->bb vs QCD; b vs c is where lifetime and vertex mass
differ physically (c tau 455-490 um for B hadrons vs 123-312 um for D hadrons; b vertex mass up to ~5 GeV, c below ~2 GeV).

Sample (random rows, never first-k)
  JetClass test_20M HToBB_100, HToCC_100, ZJetsToNuNu_100 (QCD), 2,000 random rows each (rng 4401 / 4402 / 4403). Every variant,
  identity included, is rebuilt with part_core.build (eta = jet_eta + sign(jet_eta) deta); four-momenta are never changed, only the
  impact-parameter tuple t = (d0val, d0err, dzval, dzerr) of charged tracks.
Definitions
  charged track: real particle with charge != 0. S = |d0val| / d0err (transverse impact-parameter significance).
  displaced track: charged with S > 2; prompt: charged with S <= 2. n(S>s): number of charged tracks with S > s.
  S bins: [0,1], (1,2], (2,3], (3,5], (5,10], (10, inf) (upper edge included, so n(S>s) is a sum of bins).
  subjets: two exclusive-kT subjets (h41_colour_flow.ktn_batch). Displaced-system mass of subjet k: m_D,k = invariant mass of the
  summed stored four-momenta of the displaced tracks in k. Vertex-mass proxy m_SV = max_k m_D,k; its subjet is s*.
Edits
  L(lam)   d0val and dzval of every displaced track multiplied by lam (errors kept): S -> lam S. Changes n(S>s).
  Lc(lam)  count-preserving lifetime scaling: displaced tracks only; in a finite bin (lo, hi]: u = ln(S/lo)/ln(hi/lo),
           u' = u^(1/lam), S' = lo (hi/lo)^u'; in (10, inf): S' = 10 + lam (S - 10). d0val, dzval multiplied by S'/S. Monotone, every
           track stays in its S bin, so every n(S>s) is unchanged; lam < 1 moves values toward the lower bin edge (the open tail's excess
           over 10 is scaled by lam exactly), lam > 1 toward the upper edge. Tracks whose float32 S would leave the bin are left unedited (counted).
  Vm / Vp  inside subjet s*, k in {1, 2} displaced tracks exchange t with k prompt tracks of the same subjet (among its 15 hardest
           prompt tracks), chosen so that m_D,s* decreases by >= 1 GeV (Vm) or increases by 1-4 GeV (Vp); among valid exchanges the
           one with the smallest |ln(sum pT in / sum pT out)| is taken. The jet's and each subjet's displaced counts and the S
           multiset are unchanged (asserted).
  C1m / C1p  control: same subjet, same k as Vm / Vp, exchange with |dm_D,s*| <= 5 % of m_D,s* (smallest pT mismatch).
  Cr       control: k (as Vm) random displaced <-> random prompt exchanges inside s* (rng 4410), mass change unconstrained.
Models: ParT_full, PN_full (read d0/dz); ParT_kinpid, PN_kinpid, ParT_kin, PN_kin (no IP inputs: response 0 by construction);
  ParT_perm1..6 = ParT_full with every weight tensor's entries permuted (torch seeds 5..10), the null band.
Scores (10-class logits L): s_bc = log(p_bb/p_cc) = L_Hbb - L_Hcc; s_bq = log(p_bb/p_QCD) = L_Hbb - L_QCD; bb-vs-cc call = s_bc > 0.
References (sklearn, trained on JetClass train_100M HToBB_000 / HToCC_000, NREF random rows each, rng 4404 / 4405, label bb = 1):
  LR_count (n(S>2), n(S>3)); LR_life (n(S>2), median ln S and max ln S of displaced tracks); LR_vm (n(S>2), m_SV);
  GBDT (HistGradientBoosting) on n(S>1,2,3,5,10), median/max ln S, sum ln(S/2), m_SV, jet-level displaced mass. Each is scored on
  the same edited tuples (features recomputed); its change in log-odds is the response predicted by that hypothesis.
Statistics: per (model, edit, class, score) mean with 95 % bootstrap over jets (B resamples), two-sided sign-flip permutation p,
  Holm over all tested cells (ParT_full, PN_full x edits/contrasts x class x score); McNemar on bb-vs-cc calls (edit vs identity for
  L/Lc, V vs C1 for V); permuted band = [min, max] of the 6 permuted-network means.
Run: ./sync.sh && ./part_job.sh d-h44-run 'python -m interp.h44_bc_lifetime'   (H44_SMOKE=1: 300 per class, 3,000 ref jets)
Outputs: $AI_RESULTS/physics/h44_bc_lifetime.json, h44_perjet.npz, h44_jet_edit.png, h44_response.png
"""
import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
SMOKE = bool(os.environ.get('H44_SMOKE'))
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
TAG = 'h44_smoke' if SMOKE else 'h44'
PER = 300 if SMOKE else 2000
NREF = 3000 if SMOKE else 20000
B = 300 if SMOKE else 2000
NPERM = 500 if SMOKE else 5000
NPN = 2 if SMOKE else 6
LAMS = (0.4, 0.7, 1.4, 2.0)
EDGES = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 10.0, np.inf])
IPK = ('d0val', 'd0err', 'dzval', 'dzerr')
QCD, HBB, HCC = 0, 1, 2
CLS = (('Hbb', 'HToBB', 4401, HBB), ('Hcc', 'HToCC', 4402, HCC), ('QCD', 'ZJetsToNuNu', 4403, QCD))
TEST = Path('/j-jepa-vol/JetClass/Pythia/test_20M')
TRAIN = Path('/j-jepa-vol/JetClass/Pythia/train_100M')
FLAGS = ('isChargedHadron', 'isNeutralHadron', 'isPhoton', 'isElectron', 'isMuon')
VARS = tuple(f'L{l}' for l in LAMS) + tuple(f'Lc{l}' for l in LAMS) + ('Vm', 'C1m', 'Vp', 'C1p', 'Cr')
TESTED_NETS = ('ParT_full', 'PN_full')
ZERO_NETS = ('ParT_kinpid', 'PN_kinpid', 'ParT_kin', 'PN_kin')
PERM_NETS = tuple(f'ParT_perm{i}' for i in range(1, NPN + 1))
NETS = TESTED_NETS + ZERO_NETS + PERM_NETS
REFS = ('LR_count', 'LR_life', 'LR_vm', 'GBDT')


# ================================================================================================================ data
def load_rows(path, n, seed, chunk=20000):
    import uproot
    from interp.part_core import read_root, sub, cat
    N = uproot.open(path)['tree'].num_entries
    if SMOKE: N = min(N, 40000)
    rows = np.sort(np.random.default_rng(seed).choice(N, n, replace=False)); parts = []
    for s in range(0, N, chunk):
        r = rows[(rows >= s) & (rows < s + chunk)]
        if len(r): parts.append(sub(read_root(path, start=s, stop=min(s + chunk, N)), r - s))
    log(f'{path.name}: {n} random rows of {N} (rng {seed})')
    return cat(parts), dict(file=path.name, rows_in_file=int(N), drawn=int(n), seed=seed)


def add_kinpid():
    from jetdistill.part import network as NW
    if 'kinpid' not in NW.FEATURES:
        NW.FEATURES['kinpid'] = NW.KIN + [(k, None, 1, None) for k in ('charge',) + FLAGS] + [('deta', None, 1, None), ('dphi', None, 1, None)]
    return NW


def sig(J, ip=None):
    """charged mask and S = |d0val|/d0err (0 for non-charged); ip overrides the tuple arrays"""
    d0 = (ip or J)['d0val']; de = (ip or J)['d0err']
    ch = (J['mask'] > 0) & (J['charge'] != 0)
    return ch, np.where(ch, np.abs(d0) / np.maximum(de, 1e-6), 0.0).astype(np.float64)


def subjet_labels(J):
    from interp.h41_colour_flow import ktn_batch
    m = J['mask'] > 0; nn = m.sum(1).astype(np.int64)
    P4 = (np.stack([J['px'], J['py'], J['pz'], J['energy']], -1) * m[..., None]).astype(np.float64)
    return ktn_batch(np.ascontiguousarray(P4), nn, 2), P4


def m4(v):
    return np.sqrt(np.maximum(v[..., 3] ** 2 - (v[..., :3] ** 2).sum(-1), 0))


def feats(J, LAB, P4, ip=None):
    """per-jet observables used by the references and the edit bookkeeping"""
    ch, S = sig(J, ip); d = ch & (S > 2); out = {f'n{t}': (ch & (S > t)).sum(1).astype(np.float64) for t in (1, 2, 3, 5, 10)}
    lS = np.where(d, np.log(np.maximum(S, 1e-12)), np.nan); has = d.any(1); l2 = np.log(2.0)
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out['medlogS'] = np.where(has, np.nanmedian(np.where(has[:, None], lS, 0.0), 1), l2)
            out['maxlogS'] = np.where(has, np.nanmax(np.where(has[:, None], lS, 0.0), 1), l2)
    out['sumlogS'] = np.where(d, np.log(np.maximum(S, 1e-12) / 2.0), 0.0).sum(1)
    L = LAB.shape[1]; dd = d[:, :L]
    mk = np.stack([m4((P4[:, :L] * (dd & (LAB == k))[..., None]).sum(1)) for k in range(2)], 1)
    out['msv'] = mk.max(1); out['sstar'] = mk.argmax(1); out['mjet'] = m4((P4 * d[..., None]).sum(1)); out['mk'] = mk
    out['nk'] = np.stack([(dd & (LAB == k)).sum(1) for k in range(2)], 1)
    return out


REF_COLS = dict(LR_count=('n2', 'n3'), LR_life=('n2', 'medlogS', 'maxlogS'), LR_vm=('n2', 'msv'),
                GBDT=('n1', 'n2', 'n3', 'n5', 'n10', 'medlogS', 'maxlogS', 'sumlogS', 'msv', 'mjet'))


def X_of(F, name):
    return np.column_stack([F[c] for c in REF_COLS[name]])


# ================================================================================================================ edits
def lc_factor(S, d, lam):
    """per-track multiplicative factor of the count-preserving lifetime map (1 for non-displaced tracks)"""
    f = np.ones_like(S); b = np.clip(np.searchsorted(EDGES, S, side='left') - 1, 0, len(EDGES) - 2)
    lo = EDGES[b]; hi = EDGES[b + 1]; fin = d & np.isfinite(hi); tail = d & ~np.isfinite(hi)
    with np.errstate(all='ignore'):
        u = np.where(fin, np.log(np.where(fin, S, 1) / np.where(fin, lo, 1)) / np.log(np.where(fin, hi, 2) / np.where(fin, lo, 1)), 0)
        S2 = np.where(fin, lo * (np.where(fin, hi, 2) / np.where(fin, lo, 1)) ** (np.clip(u, 0, 1) ** (1.0 / lam)), S)
        S2 = np.where(tail, 10.0 + lam * (np.where(tail, S, 10) - 10.0), S2)
        f = np.where(d & (S > 0), S2 / np.where(S > 0, S, 1), 1.0)
    return f, b


def make_edits(J, LAB, P4, F0):
    N = len(J['label']); ch, S = sig(J); d = ch & (S > 2); pt = np.hypot(J['px'], J['py'])
    E = {}; info = {}
    for lam in LAMS:
        e = {k: J[k].copy() for k in IPK}
        e['d0val'] = np.where(d, J['d0val'] * lam, J['d0val']).astype(np.float32); e['dzval'] = np.where(d, J['dzval'] * lam, J['dzval']).astype(np.float32)
        E[f'L{lam}'] = e
        f, b = lc_factor(S, d, lam); e = {k: J[k].copy() for k in IPK}
        e['d0val'] = (J['d0val'] * f).astype(np.float32); e['dzval'] = (J['dzval'] * f).astype(np.float32)
        _, S2 = sig(J, e); b2 = np.clip(np.searchsorted(EDGES, S2, side='left') - 1, 0, len(EDGES) - 2); bad = d & (b2 != b)
        for k in ('d0val', 'dzval'): e[k] = np.where(bad, J[k], e[k]).astype(np.float32)
        E[f'Lc{lam}'] = e; info[f'Lc{lam}_reverted_tracks'] = int(bad.sum())
    for v in ('Vm', 'C1m', 'Vp', 'C1p', 'Cr'): E[v] = {k: J[k].copy() for k in IPK}
    bk = {k: np.zeros(N) for k in ('kVm', 'kVp', 'dmVm', 'dmC1m', 'dmVp', 'dmC1p', 'dmCr', 'misVm', 'misC1m', 'misVp', 'misC1p', 'misCr')}
    okm = np.zeros(N, bool); okp = np.zeros(N, bool); rng = np.random.default_rng(4410); L = LAB.shape[1]

    def swap(v, j, a, b_):
        for k in IPK:
            A = E[v][k]; t = A[j, a].copy(); A[j, a] = A[j, b_]; A[j, b_] = t

    for j in range(N):
        s = F0['sstar'][j]; idx = np.arange(L)
        Dj = idx[d[j, :L] & (LAB[j] == s)]; Pj = idx[ch[j, :L] & (S[j, :L] <= 2) & (LAB[j] == s)]
        if len(Dj) == 0 or len(Pj) == 0: continue
        Pj = Pj[np.argsort(-pt[j, Pj], kind='stable')][:15]; sD = P4[j, Dj].sum(0); m0 = m4(sD)
        found = {}
        for k in (1, 2):
            if len(Dj) < k or len(Pj) < k: break
            cd = np.array(list(combinations(range(len(Dj)), k))); cp = np.array(list(combinations(range(len(Pj)), k)))
            dv = P4[j, Dj][cd].sum(1); pv = P4[j, Pj][cp].sum(1); dp = pt[j, Dj][cd].sum(1); pp = pt[j, Pj][cp].sum(1)
            dm = m4(sD[None, None] - dv[:, None] + pv[None]) - m0; mis = np.abs(np.log(pp[None] / np.maximum(dp[:, None], 1e-9)))
            cond = dict(Vm=dm <= -1.0, Vp=(dm >= 1.0) & (dm <= 4.0), C1=np.abs(dm) <= 0.05 * m0)
            for nm, c in cond.items():
                if c.any():
                    mm = np.where(c, mis, np.inf); a, b_ = np.unravel_index(np.argmin(mm), mm.shape)
                    found[(nm, k)] = (Dj[cd[a]], Pj[cp[b_]], float(dm[a, b_]), float(mis[a, b_]))
        for V, C in (('Vm', 'C1m'), ('Vp', 'C1p')):
            for k in (1, 2):
                if (V, k) in found and ('C1', k) in found:
                    a, b_, dmv, mv = found[(V, k)]; swap(V, j, a, b_); bk[f'dm{V}'][j] = dmv; bk[f'mis{V}'][j] = mv; bk[f'k{V}'][j] = k
                    a, b_, dmc, mc = found[('C1', k)]; swap(C, j, a, b_); bk[f'dm{C}'][j] = dmc; bk[f'mis{C}'][j] = mc
                    (okm if V == 'Vm' else okp)[j] = True
                    if V == 'Vm':
                        a = rng.choice(Dj, k, replace=False); b_ = rng.choice(Pj, k, replace=False); swap('Cr', j, a, b_)
                        bk['dmCr'][j] = m4(sD - P4[j, a].sum(0) + P4[j, b_].sum(0)) - m0
                        bk['misCr'][j] = abs(np.log(pt[j, b_].sum() / max(pt[j, a].sum(), 1e-9)))
                    break
    return E, info, bk, okm, okp


def variant_dict(J, Ev):
    from interp.part_core import build, kin
    ex = dict(J); ex.update(Ev); return build(J, *kin(J), extra=ex)


# ================================================================================================================ models
def load_net(name):
    import torch
    from interp.part_core import ParTNetwork
    from interp.pn_net import load_particlenet
    NW = add_kinpid(); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name.startswith('ParT_perm'):
        net = ParTNetwork('full'); g = torch.Generator().manual_seed(4 + int(name[-1]))
        with torch.no_grad():
            for p in net.model.parameters():
                if p.numel() > 1: p.copy_(p.flatten()[torch.randperm(p.numel(), generator=g).to(p.device)].view_as(p))
        return ('part', 'full', net)
    kind, n = name.split('_')
    if kind == 'ParT': return ('part', n, ParTNetwork(n))
    pnd = Path(os.environ.get('JETDISTILL_MODELS', '/j-jepa-vol/aaron-interp/models/part')).parent / 'particlenet'
    return ('pn', n, load_particlenet(pnd / f'ParticleNet_{n}.pt', len(NW.FEATURES[n]), dev))


def logits(M, Jd):
    from interp.part_core import inputs
    from interp.p1v_verify_part import fwd
    from interp.h1_irc_panel import pn_logits
    kind, n, net = M
    return np.asarray(fwd(net, *inputs(Jd, n))[1] if kind == 'part' else pn_logits(net, Jd, n), np.float64)


def lo(p):
    p = np.clip(p, 1e-7, 1 - 1e-7); return np.log(p / (1 - p))


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


def mcnemar(a, b):
    from scipy.stats import binomtest
    n01 = int((a & ~b).sum()); n10 = int((~a & b).sum())
    return dict(bb_to_cc=n01, cc_to_bb=n10, p=float(binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0)


def auc(s, y):
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return clean(o.tolist())
    return o


# ================================================================================================================ main
def main():
    from interp.part_core import cat, read_root
    RES.mkdir(parents=True, exist_ok=True)
    out = dict(node='H44', smoke=SMOKE, per_class=PER, n_ref=NREF, B=B, n_perm=NPERM, lambdas=LAMS, s_bins=EDGES.tolist())
    parts = []; out['sample'] = {}
    for c, pre, seed, lab in CLS:
        Jc, inf = load_rows(TEST / f'{pre}_100.root', PER, seed); parts.append(Jc); out['sample'][c] = inf
        assert np.all(Jc['label'] == lab), f'{c} labels'
    J = cat(parts); y = J['label'].astype(int); N = len(y)
    LAB, P4 = subjet_labels(J); F0 = feats(J, LAB, P4); log(f'sample {N} jets; kT-2 subjets and observables done')
    out['observables_unedited'] = {c: {k: float(np.mean(F0[k][y == lab])) for k in ('n1', 'n2', 'n3', 'n5', 'n10', 'medlogS', 'maxlogS', 'msv', 'mjet')}
                                   for c, _, _, lab in CLS}
    log('observables', json.dumps(out['observables_unedited']))
    # ---- edits
    E, out['edit_info'], bk, okm, okp = make_edits(J, LAB, P4, F0)
    FE = {v: feats(J, LAB, P4, E[v]) for v in VARS}; ch, S0 = sig(J); S0s = np.sort(S0, 1)
    chk = {}
    for v in VARS:
        f = FE[v]; c = {}
        for c_, _, _, lab in CLS:
            m = y == lab
            c[c_] = {k: float(np.mean(f[k][m] - F0[k][m])) for k in ('n1', 'n2', 'n3', 'n5', 'n10', 'medlogS', 'maxlogS', 'msv', 'mjet')}
        if v.startswith('Lc') or v[0] in 'VC':
            for t in (1, 2, 3, 5, 10): assert np.array_equal(f[f'n{t}'], F0[f'n{t}']), f'{v} changed n(S>{t})'
        if v[0] in 'VC':
            assert np.allclose(np.sort(sig(J, E[v])[1], 1), S0s), f'{v} changed the S multiset'
            assert np.array_equal(f['nk'], F0['nk']), f'{v} changed a subjet displaced count'
        chk[v] = c
    out['edit_observable_change'] = chk
    out['V_sets'] = {c: dict(n_Vm=int((okm & (y == lab)).sum()), n_Vp=int((okp & (y == lab)).sum())) for c, _, _, lab in CLS}
    setk = lambda k: okp if k.endswith(('Vp', 'C1p')) else okm
    out['V_book'] = {c: {k: (float(np.mean(v[setk(k) & (y == lab)])) if (setk(k) & (y == lab)).any() else None) for k, v in bk.items()}
                     for c, _, _, lab in CLS}
    log('V sets', out['V_sets']); log('V book', json.dumps(out['V_book'])); log('Lc reverted', out['edit_info'])
    # ---- networks
    Jid = variant_dict(J, {k: J[k] for k in IPK}); Jvar = {v: variant_dict(J, E[v]) for v in VARS}; log('variants built')
    SC = {}; out['unedited'] = {}; hb = np.isin(y, [HBB, HCC])
    for name in NETS:
        M = load_net(name); Lid = logits(M, Jid); SC[(name, 'id')] = Lid
        u = dict(acc_10class=float((Lid.argmax(1) == y).mean()),
                 acc_10class_by_class={c: float((Lid.argmax(1) == lab)[y == lab].mean()) for c, _, _, lab in CLS},
                 acc_bb_vs_cc=float(((Lid[hb, HBB] > Lid[hb, HCC]) == (y[hb] == HBB)).mean()),
                 auc_bb_vs_cc=auc(Lid[hb, HBB] - Lid[hb, HCC], (y[hb] == HBB).astype(int)),
                 auc_bb_vs_qcd=auc(Lid[y != HCC, HBB] - Lid[y != HCC, QCD], (y[y != HCC] == HBB).astype(int)))
        out['unedited'][name] = u; log(f'{name}: ' + json.dumps(u))
        for v in VARS: SC[(name, v)] = logits(M, Jvar[v])
        if name == 'ParT_full':
            out['input_path_check'] = float(np.abs(logits(M, J) - Lid).max()); log('raw vs build max |dlogit|', out['input_path_check'])
        del M
        try:
            import torch; torch.cuda.empty_cache()
        except Exception: pass
    out['zero_check'] = {n: float(max(np.abs(SC[(n, v)] - SC[(n, 'id')]).max() for v in VARS)) for n in ZERO_NETS}
    log('zero-model max |dlogit|', out['zero_check'])
    # ---- references
    Fr, yr = [], []; out['ref_rows'] = {}
    for pre, lab, seed in (('HToBB', 1, 4404), ('HToCC', 0, 4405)):
        Jr, inf = load_rows(TRAIN / f'{pre}_000.root', NREF, seed); out['ref_rows'][pre] = inf
        LABr, P4r = subjet_labels(Jr); Fr.append(feats(Jr, LABr, P4r)); yr.append(np.full(len(Jr['label']), lab))
    Fr = {k: np.concatenate([f[k] for f in Fr]) for k in REF_COLS['GBDT']}; yr = np.concatenate(yr)
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    tr = np.random.default_rng(4406).random(len(yr)) < 0.8; REF = {}; out['ref_info'] = {}
    for r in REFS:
        mdl = (HistGradientBoostingClassifier(max_iter=300, random_state=0) if r == 'GBDT' else
               make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))).fit(X_of(Fr, r)[tr], yr[tr])
        REF[r] = mdl; out['ref_info'][r] = dict(features=REF_COLS[r], heldout_acc=float(mdl.score(X_of(Fr, r)[~tr], yr[~tr])),
                                                heldout_auc=auc(mdl.predict_proba(X_of(Fr, r)[~tr])[:, 1], yr[~tr]))
        if r != 'GBDT':
            lr = mdl[-1]; out['ref_info'][r]['coef_std_units'] = lr.coef_[0].tolist()
        log(f'ref {r}: ' + json.dumps(out['ref_info'][r]))
    RL = {}
    for r in REFS:
        RL[(r, 'id')] = lo(REF[r].predict_proba(X_of(F0, r))[:, 1])
        for v in VARS: RL[(r, v)] = lo(REF[r].predict_proba(X_of(FE[v], r))[:, 1])
        out['ref_info'][r]['acc_bb_vs_cc_on_sample'] = float(((RL[(r, 'id')][hb] > 0) == (y[hb] == HBB)).mean())

    # ---- statistics
    rng = np.random.default_rng(4499)
    score = dict(bc=lambda L: L[:, HBB] - L[:, HCC], bq=lambda L: L[:, HBB] - L[:, QCD])

    def resp(m, v, sc):
        if m in REFS:
            if sc != 'bc': return None
            return RL[(m, v)] - RL[(m, 'id')]
        return score[sc](SC[(m, v)]) - score[sc](SC[(m, 'id')])

    CON = {v: (lambda m, sc, v=v: resp(m, v, sc)) for v in VARS}
    CON['Vm-C1m'] = lambda m, sc: (None if resp(m, 'Vm', sc) is None else resp(m, 'Vm', sc) - resp(m, 'C1m', sc))
    CON['Vp-C1p'] = lambda m, sc: (None if resp(m, 'Vp', sc) is None else resp(m, 'Vp', sc) - resp(m, 'C1p', sc))
    CON['Cr-C1m'] = lambda m, sc: (None if resp(m, 'Cr', sc) is None else resp(m, 'Cr', sc) - resp(m, 'C1m', sc))
    SETOF = lambda cn: okm if cn in ('Vm', 'C1m', 'Cr', 'Vm-C1m', 'Cr-C1m') else (okp if cn in ('Vp', 'C1p', 'Vp-C1p') else np.ones(N, bool))
    TESTED = [f'L{l}' for l in LAMS] + [f'Lc{l}' for l in LAMS] + ['Vm-C1m', 'Vp-C1p', 'Cr-C1m']
    res, cells = {}, []
    for c, _, _, lab in CLS:
        for sc in ('bc', 'bq'):
            for cn, fn in CON.items():
                sm = SETOF(cn) & (y == lab)
                for m in NETS + REFS:
                    x = fn(m, sc)
                    if x is None: continue
                    x = x[sm]; mu, ci = boot(x, rng); cell = dict(mean=mu, ci=ci, n=int(sm.sum()))
                    if m in TESTED_NETS and cn in TESTED:
                        cell['p_perm'] = perm2(x, rng); cells.append(f'{c}|{sc}|{cn}|{m}')
                    res[f'{c}|{sc}|{cn}|{m}'] = cell
                pm = [res[f'{c}|{sc}|{cn}|{m}']['mean'] for m in PERM_NETS]
                res[f'{c}|{sc}|{cn}|perm_band'] = dict(min=float(np.min(pm)), max=float(np.max(pm)), mean=float(np.mean(pm)), sd=float(np.std(pm)))
            for m in TESTED_NETS:
                call = lambda v: SC[(m, v)][:, HBB] > SC[(m, v)][:, HCC]
                ym = y == lab
                for v in VARS:
                    if v.startswith('L'): res[f'{c}|{m}|mcnemar|{v}_vs_id'] = mcnemar(call('id')[ym], call(v)[ym])
                for V, C, ok in (('Vm', 'C1m', okm), ('Vp', 'C1p', okp)):
                    s_ = ok & ym; res[f'{c}|{m}|mcnemar|{V}_vs_{C}'] = mcnemar(call(C)[s_], call(V)[s_])
                res[f'{c}|{m}|bbcall_rate'] = {v: float(call(v)[ym].mean()) for v in ('id',) + VARS}
            # Lc / L ratio per lambda (bc score, tested nets) with bootstrap
            if True:
                for m in TESTED_NETS + REFS:
                    for l in LAMS:
                        a = CON[f'Lc{l}'](m, 'bc'); b_ = CON[f'L{l}'](m, 'bc')
                        if a is None or sc != 'bc': continue
                        a, b_ = a[y == lab], b_[y == lab]
                        if abs(b_.mean()) < 1e-9: continue
                        ii = rng.integers(0, len(a), (B, len(a))); r_ = a[ii].mean(1) / b_[ii].mean(1)
                        res[f'{c}|bc|ratio_Lc{l}_to_L{l}|{m}'] = dict(ratio=float(a.mean() / b_.mean()), ci=np.percentile(r_, [2.5, 97.5]).tolist())
    adj = holm([res[c]['p_perm'] for c in cells])
    for c, a in zip(cells, adj): res[c]['p_holm'] = float(a)
    out['holm_family'] = dict(n_tests=len(cells)); out['results'] = res
    for c, _, _, lab in CLS:
        for cn in TESTED:
            s = '  '.join(f'{m} {res[f"{c}|bc|{cn}|{m}"]["mean"]:+.3f}' for m in TESTED_NETS + REFS if f'{c}|bc|{cn}|{m}' in res)
            pb = res[f'{c}|bc|{cn}|perm_band']
            log(f'{c} bc {cn}: {s}  perm [{pb["min"]:+.3f}, {pb["max"]:+.3f}]  ParT holm {res[f"{c}|bc|{cn}|ParT_full"]["p_holm"]:.2g}')
    (RES / f'{TAG}_bc_lifetime.json').write_text(json.dumps(clean(out), indent=1))
    keep = dict(y=y, okm=okm, okp=okp, **{f'bk_{k}': v for k, v in bk.items()},
                **{f'F0_{k}': F0[k] for k in ('n2', 'n3', 'medlogS', 'maxlogS', 'msv', 'sstar')})
    for (m, v), L in SC.items():
        if m in TESTED_NETS or m == 'ParT_perm1': keep[f'L|{m}|{v}'] = L.astype(np.float32)
    for (m, v), L in RL.items(): keep[f'ref|{m}|{v}'] = L.astype(np.float32)
    np.savez_compressed(RES / f'{TAG}_perjet.npz', **keep); log('saved')
    try:
        fig_jet(J, E, LAB, F0, FE, okm, y)
    except Exception as e: log('jet figure failed', repr(e))
    try:
        fig_resp(out)
    except Exception as e: log('response figure failed', repr(e))


# ================================================================================================================ figures
def fig_jet(J, E, LAB, F0, FE, okm, y):
    import matplotlib
    matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ch, S = sig(J); cand = np.flatnonzero((y == HBB) & okm & (F0['n2'] >= 6) & (F0['n2'] <= 10) & (F0['n10'] >= 2))
    if not len(cand): cand = np.flatnonzero((y == HBB) & okm)
    j = cand[np.argmin(np.abs(F0['msv'][cand] - np.median(F0['msv'][cand])))]
    panels = [('unedited', None), ('Lc(0.4): S shrunk inside each S bin', 'Lc0.4'), ('Vm: IP tuples exchanged to lower m_SV', 'Vm')]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2), sharex=True, sharey=True); m = J['mask'][j] > 0
    for ax, (tt, v) in zip(axs, panels):
        Sv = S[j] if v is None else sig(J, E[v])[1][j]; Fv = F0 if v is None else FE[v]
        ax.scatter(J['deta'][j, m & ~ch[j]], J['dphi'][j, m & ~ch[j]], s=6, c='#bbbbbb', label='neutral particle')
        pr = m & ch[j] & (Sv <= 2); ds = m & ch[j] & (Sv > 2)
        ax.scatter(J['deta'][j, pr], J['dphi'][j, pr], s=12, facecolors='none', edgecolors='#555555', label='prompt track (S <= 2)')
        sc = ax.scatter(J['deta'][j, ds], J['dphi'][j, ds], s=10 + 120 * np.log10(Sv[ds] / 2 + 1), c=np.log10(Sv[ds]), cmap='viridis', vmin=np.log10(2), vmax=3,
                        edgecolors='k', linewidths=0.5, label='displaced track (S > 2),\narea grows with log S')
        ax.set_title(f'{tt}\nn(S>2) = {int(Fv["n2"][j])}, median S = {np.exp(Fv["medlogS"][j]):.1f}, max S = {np.exp(Fv["maxlogS"][j]):.0f}\n'
                     f'displaced-system mass m_SV = {Fv["msv"][j]:.2f} GeV', fontsize=9)
        ax.set_xlabel('Δη to jet axis')
    axs[0].set_ylabel('Δφ to jet axis'); axs[0].legend(fontsize=7, loc='lower left', frameon=False)
    cb = fig.colorbar(sc, ax=axs, shrink=0.8, pad=0.01); cb.set_label('log10 S (transverse impact-parameter significance)')
    fig.suptitle(f'One JetClass H→bb jet (pT {J["jet_pt"][j]:.0f} GeV): the edits change the impact parameters of tracks, never their momenta', fontsize=11, y=1.07)
    fig.savefig(RES / f'{TAG}_jet_edit.png', dpi=120, bbox_inches='tight'); log('jet figure saved')


def fig_resp(out):
    import matplotlib
    matplotlib.use('Agg'); import matplotlib.pyplot as plt
    R = out['results']; cons = [f'L{l}' for l in LAMS] + [f'Lc{l}' for l in LAMS] + ['Vm-C1m', 'Vp-C1p']
    labs = [f'all IP x{l}' for l in LAMS] + [f'in-bin x{l}\n(counts fixed)' for l in LAMS] + ['mass -\nminus control', 'mass +\nminus control']
    ms = [('ParT_full', 'ParT_full', '#2f5d8a'), ('PN_full', 'ParticleNet_full', '#c0603a'), ('LR_count', 'fit on counts', '#9a9a9a'),
          ('LR_life', 'fit on counts + S size', '#5a9a5a'), ('LR_vm', 'fit on counts + m_SV', '#b5a33a')]
    fig, ax = plt.subplots(figsize=(14, 4.8)); w = 0.16
    for i, (m, lab, col) in enumerate(ms):
        mu = np.array([R[f'Hbb|bc|{c}|{m}']['mean'] for c in cons]); ci = np.array([R[f'Hbb|bc|{c}|{m}']['ci'] for c in cons])
        ax.bar(np.arange(len(cons)) + (i - 2) * w, mu, w, color=col, label=lab, yerr=[mu - ci[:, 0], ci[:, 1] - mu], capsize=2)
    for k, c in enumerate(cons):
        pb = R[f'Hbb|bc|{c}|perm_band']; ax.plot([k - 0.45, k + 0.45], [pb['min']] * 2, 'k:', lw=0.8); ax.plot([k - 0.45, k + 0.45], [pb['max']] * 2, 'k:', lw=0.8)
    ax.axhline(0, color='k', lw=0.6); ax.set_xticks(range(len(cons))); ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel('change in log(p_bb / p_cc)\non H→bb jets (negative = toward cc)')
    ax.set_title(f'H→bb jets (N = {out["per_class"]}): response to impact-parameter edits; dotted = range of {NPN} weight-permuted ParT', fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1.0)); fig.tight_layout()
    fig.savefig(RES / f'{TAG}_response.png', dpi=130); log('response figure saved')


def fig_only():
    """H44_FIG=1: redraw both figures from the same sample (no networks)"""
    from interp.part_core import cat
    J = cat([load_rows(TEST / f'{pre}_100.root', PER, seed)[0] for _, pre, seed, _ in CLS]); y = J['label'].astype(int)
    LAB, P4 = subjet_labels(J); F0 = feats(J, LAB, P4); E, _, _, okm, _ = make_edits(J, LAB, P4, F0)
    FE = {v: feats(J, LAB, P4, E[v]) for v in ('Lc0.4', 'Vm')}; fig_jet(J, E, LAB, F0, FE, okm, y)
    fig_resp(json.loads((RES / f'{TAG}_bc_lifetime.json').read_text()))


if __name__ == '__main__':
    fig_only() if os.environ.get('H44_FIG') else main()
