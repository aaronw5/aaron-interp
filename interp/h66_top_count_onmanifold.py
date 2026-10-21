"""H66 (loop D, claim D24 v2): top-tagger object-count response on generated jets matched to Top Landscape, with the data's
conditional count likelihood ratio LR(N | m, tau32, mW) as the physics reference.

Question: the H61 / H61v Delphes jets (Pythia 8.317, MPI off, Delphes 3.5.1 ATLAS card) are separable from Top Landscape test jets
(C2ST AUC 0.74).  After reweighting them to Top Landscape (density-ratio weights, C2ST target <= 0.6 on a held-out half), does the
paired hadronisation change Hz+ (about +7 % reconstructed objects at fixed m, tau32, mW) still lower the top score of the Top
Landscape taggers?  And on real Top Landscape jets, is the sign of the networks' response per added object the sign of the
data's own d log-odds / dN at fixed (m, tau32, mW)?

Samples (all existing, no new generation):
  half A = data/top/gen_h61  (workers 6101-6110), V0 (Monash) and Hzp (aLund 0.79, bLund 0.88, sigma 0.247 GeV), event-paired
  half B = data/top/gen_h61v (workers 6201-6210), same settings, independent events
  cnt    = per pair, the V0 jet brought to exactly the Hz+ jet's object count: d = N_Hz+ - N_V0; d > 0: the d softest objects
           each split into two pT/2 halves +-0.005 apart in (eta, phi) (H61.split_soft); d < 0: the |d| nearest pairs in the
           softer half merged (H61.merge_soft); the count part of the Hz+ shift.  rest = Hz+ - cnt.
  TL     = Top Landscape test rows, rng 6611, 20k per class, alternately assigned to T1 (weight fitting) and T2 (C2ST check)
  E_*    = edits on 5k + 5k of the T2 rows (every second row): split_rand k = 2, 4, 8 (the first k of 8 constituents drawn at random,
           each replaced by two pT/2 halves +-0.005 apart in a random direction: N + k, jet four-momentum and every IRC-safe
           observable fixed to O(1e-5)); jit8 (the same 8 constituents moved by 0.005 in the same direction, pT kept: N fixed);
           ss8 (split of the 8 softest constituents)
  TLtr   = Top Landscape train rows, rng 6612, 60k per class, (pT, deta, dphi) from kin3 (128 hardest, float16) -> m, tau32, mW,
           tau21, mSD; N = jet_nparticles; the data reference fits.
Models: Top Landscape ParT, ParticleNet, LorentzNet (batch 32, key-hashed order), PFN, EFN; JetClass ParT_kin, PN_kin; six
weight-permuted ParT copies (seeds 0-5).  Score s = top log-odds.
Steps: score (GPU, env-part) -> h66_scores.npz, h66_tltrain.npz;  stats (CPU) -> h66_run.json, h66_count_onmanifold.png
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
V = Path('/j-jepa-vol/aaron-interp')
RES = Path(os.environ.get('AI_RESULTS', V / 'results')) / 'physics'
HALVES = {'A': V / 'data' / 'top' / 'gen_h61', 'B': V / 'data' / 'top' / 'gen_h61v'}
KS = (2, 4, 8)
NPERM = 6
NETS = ['ParT_top', 'PN_top', 'LNet_top', 'PFN_top', 'EFN_top', 'ParT_kin', 'PN_kin']
TESTNETS = ['ParT_top', 'PN_top', 'LNet_top', 'PFN_top', 'ParT_kin', 'PN_kin']
PERMS = [f'perm{i}' for i in range(NPERM)]
WF = ['N', 'm', 'tau32', 'mW', 'pTD', 'f1', 'f2', 'f3', 'f5', 'f10', 'f20', 'f40', 'n_lt1', 'n_1_2', 'n_2_5',
      'girth', 'fr01', 'fr02', 'fr04', 'nr01', 'nr13', 'nr38', 'jpt', 'jaeta']           # weighting / main C2ST features
XF = ['tau21', 'mSD', 'm12min', 'm123']                                                   # substructure not used in the weights
SF = ['ParT_top', 'PN_top', 'PFN_top', 'EFN_top', 'LNet_top']                             # network scores (C2ST "where the nets look")


# ======================================================================================================== jets and edits
def load_gen(h, s):
    Z = np.load(HALVES[h] / f'{s}.npz'); return {k: Z[k] for k in Z.files}


def rows_from_P4(P, n):
    """list of (n, 4) pt, eta, phi, code(0) arrays, pT-sorted"""
    out = []
    for j in range(len(n)):
        p = P[j, :n[j]].astype(np.float64); pt = np.hypot(p[:, 0], p[:, 1])
        x = np.c_[pt, np.arcsinh(p[:, 2] / np.maximum(pt, 1e-12)), np.arctan2(p[:, 1], p[:, 0]), np.zeros(len(pt))]
        out.append(x[np.argsort(-x[:, 0], kind='stable')])
    return out


def P4_from_rows(L, Lmax=None):
    n = np.array([len(x) for x in L]); Lm = int(Lmax or n.max()); P = np.zeros((len(L), Lm, 4), np.float32)
    for j, x in enumerate(L):
        x = x[:Lm]; pt, eta, phi = x[:, 0], x[:, 1], x[:, 2]
        P[j, :len(x)] = np.c_[pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)]
    return P, np.minimum(n, Lm)


def pick(x, k, rng):
    n = len(x); k = int(min(k, n - 1)); return rng.choice(n, k, replace=False), rng.uniform(0, 2 * np.pi, k)


def split_at(x, ch, th, delta=0.01):
    if len(ch) == 0: return x.copy()
    a = x[ch].copy(); a[:, 0] /= 2; b = a.copy()
    a[:, 1] += 0.5 * delta * np.cos(th); a[:, 2] += 0.5 * delta * np.sin(th)
    b[:, 1] -= 0.5 * delta * np.cos(th); b[:, 2] -= 0.5 * delta * np.sin(th)
    y = np.r_[np.delete(x, ch, 0), a, b]; return y[np.argsort(-y[:, 0], kind='stable')]


def jitter_at(x, ch, th, delta=0.01):
    y = x.copy(); y[ch, 1] += 0.5 * delta * np.cos(th); y[ch, 2] += 0.5 * delta * np.sin(th)
    return y[np.argsort(-y[:, 0], kind='stable')]


def count_matched(G0, G1):
    """V0 jet brought to the paired Hz+ jet's exact object count (split softest / merge nearest soft pairs)"""
    from interp import h61_top_delphes as H61
    pos1 = {int(k): j for j, k in enumerate(G1['key'])}
    pr = np.array([(j, pos1[int(k)]) for j, k in enumerate(G0['key']) if int(k) in pos1]); rng = np.random.default_rng(6613); L = []
    for i0, i1 in pr:
        x = H61._rows(G0, i0); d = int(G1['n'][i1] - G0['n'][i0])
        L.append(H61.split_soft(x, d, rng) if d > 0 else (H61.merge_soft(x, -d) if d < 0 else x.copy()))
    return H61._pack(L, G0, pr[:, 0]), pr


# ======================================================================================================== observables
def dr_feats(P, n):
    """pT-weighted angular width (girth = sum pT dR / sum pT), pT fractions within dR < 0.1, 0.2, 0.4 of the constituent-sum axis,
    object counts in dR < 0.1, 0.1-0.3, > 0.3; jet pT and |eta|"""
    P = P.astype(np.float64); L = P.shape[1]; real = np.arange(L)[None] < n[:, None]
    S = (P * real[..., None]).sum(1); jpt = np.hypot(S[:, 0], S[:, 1]); jeta = np.arcsinh(S[:, 2] / jpt); jphi = np.arctan2(S[:, 1], S[:, 0])
    pt = np.where(real, np.hypot(P[..., 0], P[..., 1]), 0); eta = np.arcsinh(np.where(real, P[..., 2], 0) / np.maximum(pt, 1e-12))
    phi = np.arctan2(P[..., 1], P[..., 0]); dr = np.hypot(eta - jeta[:, None], (phi - jphi[:, None] + np.pi) % (2 * np.pi) - np.pi)
    s = pt.sum(1); f = lambda r: (pt * (dr < r)).sum(1) / s; c = lambda lo, hi: (real & (dr >= lo) & (dr < hi)).sum(1).astype(float)
    return dict(girth=(pt * dr).sum(1) / s, fr01=f(0.1), fr02=f(0.2), fr04=f(0.4), nr01=c(0, 0.1), nr13=c(0.1, 0.3), nr38=c(0.3, 9),
                jpt=jpt, jaeta=np.abs(jeta))


def all_obs(P, n):
    from interp import h4_top_wmass as H
    from interp import h61_top_delphes as H61
    o = H._obs(P, n); o.update(H61.pt_feats(P, n)); o.update(dr_feats(P, n)); o['N'] = n.astype(float)
    return {k: np.asarray(v, np.float64) for k, v in o.items()}


# ======================================================================================================== scoring (GPU)
def step_score(a):
    import torch
    from sklearn.metrics import roc_auc_score
    from interp import data_top as dt
    from interp import h4_top_wmass as H
    from interp import h19_wpair_cross as X
    from interp import h60_top_hadronisation as H60
    RES.mkdir(parents=True, exist_ok=True)
    sets = {}                       # name -> (P, n, y, key, order, perms?)
    for h in HALVES:
        G0, G1 = load_gen(h, 'V0'), load_gen(h, 'Hzp'); Gc, pr = count_matched(G0, G1)
        log(f'half {h}: V0 {len(G0["n"])}, Hz+ {len(G1["n"])}, pairs {len(pr)}; cnt N == Hz+ N in {np.mean(Gc["n"] == G1["n"][pr[:, 1]]):.4f} of pairs')
        for s, G in (('V0', G0), ('Hzp', G1), ('cnt', Gc)):
            P, n = H60.massless_P4(G); order = np.argsort((G['key'] * 2654435761) % 1_000_003, kind='stable')
            sets[f'{h}_{s}'] = (P, n, G['y'].astype(int), G['key'].astype(np.int64), order, True)
    # Top Landscape test rows (rng 6611): 20k per class -> T1 / T2 alternately
    y = np.load(dt.DST / 'test' / 'label.npy'); rng = np.random.default_rng(6611)
    ch = [rng.choice(np.flatnonzero(y == c), 20000, replace=False) for c in (0, 1)]
    t1 = np.sort(np.concatenate([c[0::2] for c in ch])); t2 = np.sort(np.concatenate([c[1::2] for c in ch]))
    full = H._full_test(); jn = np.load(dt.DST / 'test' / 'jet.npy')[:, 5].astype(np.int64)
    for nm, idx in (('T1', t1), ('T2', t2)):
        P = np.asarray(full[idx][..., :4], np.float32); n = np.minimum(jn[idx], P.shape[1])
        sets[nm] = (H._identity(P, n).astype(np.float32), n, y[idx].astype(int), idx.astype(np.int64),
                    np.random.default_rng(6614).permutation(len(n)), False)
    # edits on every second T2 row
    P2, n2, y2, k2 = sets['T2'][0][0::2], sets['T2'][1][0::2], sets['T2'][2][0::2], sets['T2'][3][0::2]; del full
    base = rows_from_P4(P2, n2); rng = np.random.default_rng(6615)
    picks = [pick(x, max(KS), rng) for x in base]
    Lmax = int(n2.max()) + max(KS); E = {'E_base': base}
    for k in KS: E[f'E_sr{k}'] = [split_at(x, c[:k], t[:k]) for x, (c, t) in zip(base, picks)]
    E['E_jit8'] = [jitter_at(x, c, t) for x, (c, t) in zip(base, picks)]
    E['E_ss8'] = [split_at(x, np.argsort(x[:, 0], kind='stable')[:min(8, len(x) - 1)], t) for x, (_, t) in zip(base, picks)]
    oE = np.random.default_rng(6616).permutation(len(n2))
    for nm, L in E.items():
        P, n = P4_from_rows(L, Lmax); sets[nm] = (P, n, y2, k2, oE, True)
        log(nm, 'mean N', n.mean(), 'max', n.max())
    # models
    M = X.Models(); M.massless = True; perms = []
    for i in range(NPERM):
        net = H.WeaverNet('ParT', X.RES / 'h4_ckpt_part_kin.pt', M.dev); g = torch.Generator().manual_seed(i)
        with torch.no_grad():
            for prm in net.model.parameters():
                fl = prm.data.view(-1); fl.copy_(fl[torch.randperm(fl.numel(), generator=g).to(fl.device)])
        perms.append(net)
    out = {}
    for nm, (P, n, yy, key, order, wp) in sets.items():
        t = time.time(); O = all_obs(P, n)
        for k, v in O.items(): out[f'obs|{nm}|{k}'] = v.astype(np.float32)
        out[f'y|{nm}'] = yy.astype(np.int8); out[f'key|{nm}'] = key; out[f'N|{nm}'] = n
        sc = M.score('top', P, n, X.top_base(P, n), order)
        if wp:
            for i, net in enumerate(perms): sc[f'perm{i}'] = net.score(P, n)
        for k, v in sc.items(): out[f'{nm}|{k}'] = np.asarray(v, np.float32)
        log(f'{nm}: {len(n)} jets <N> top {n[yy == 1].mean():.1f} QCD {n[yy == 0].mean():.1f} m {O["m"].mean():.1f} {time.time() - t:.0f}s | '
            + '; '.join(f'{k} acc {np.mean((v > 0) == (yy == 1)):.4f} auc {roc_auc_score(yy, v):.4f}' for k, v in sc.items()))
        np.savez_compressed(RES / 'h66_scores.npz', **out)
    log('wrote h66_scores.npz')
    step_tltrain(a)


def step_tltrain(a):
    """Top Landscape train rows (rng 6612, 60k per class): substructure from kin3 (pT, deta, dphi; massless, so m, tau32, mW are
    unchanged by the longitudinal shift deta = eta - eta_jet) and the true multiplicity N = jet_nparticles"""
    from interp import data_top as dt
    d = dt.DST / 'train'; y = np.load(d / 'label.npy'); rng = np.random.default_rng(6612)
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), a.ntrain, replace=False) for c in (0, 1)]))
    k3m = np.load(d / 'kin3.npy', mmap_mode='r'); K = np.zeros((len(idx), k3m.shape[1], 3), np.float32); B = 100_000; t = time.time()
    for s in range(0, k3m.shape[0], B):
        sel = (idx >= s) & (idx < s + B)
        if sel.any(): blk = np.asarray(k3m[s:s + B]); K[sel] = blk[idx[sel] - s]
    jet = np.load(d / 'jet.npy')[idx]; log(f'TL train {len(idx)} rows read in {time.time() - t:.0f}s')
    n = (K[..., 0] > 0).sum(1).astype(np.int64); pt, eta, phi = K[..., 0].astype(np.float64), K[..., 1].astype(np.float64), K[..., 2].astype(np.float64)
    P = (np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)], -1) * (pt > 0)[..., None]).astype(np.float32)
    from interp import h4_top_wmass as H
    o = H._obs(P, n); log(f'obs done {time.time() - t:.0f}s')
    np.savez_compressed(RES / 'h66_tltrain.npz', y=y[idx].astype(np.int8), N=jet[:, 5].astype(np.float32), n128=n, jpt=jet[:, 0], jm=jet[:, 4],
                        **{k: np.asarray(o[k], np.float32) for k in ('m', 'tau32', 'mW', 'tau21', 'mSD')})
    log('wrote h66_tltrain.npz', 'N top', jet[y[idx] == 1, 5].mean(), 'QCD', jet[y[idx] == 0, 5].mean(),
        'm check (obs vs jet_mass) median |diff|', float(np.median(np.abs(o['m'] - jet[:, 4]))))


# ======================================================================================================== statistics
def _hgb(seed=0, it=200):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=it, random_state=seed)


def c2st(Xa, Xb, wa=None, seed=0):
    """2-fold out-of-fold AUC of a HistGB classifier separating sample a (weights wa, rescaled to total len(b)) from sample b"""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    wa = np.ones(len(Xa)) if wa is None else np.asarray(wa, float)
    X = np.r_[Xa, Xb]; y = np.r_[np.ones(len(Xa)), np.zeros(len(Xb))]; sw = np.r_[wa * len(Xb) / wa.sum(), np.ones(len(Xb))]
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(2, shuffle=True, random_state=seed).split(X, y):
        p[te] = _hgb(seed).fit(X[tr], y[tr], sample_weight=sw[tr]).predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, p, sample_weight=sw))


def fit_ratio(Xs, Xt, seed=0):
    """calibrated (Platt, 3-fold) HistGB classifier source (0) vs target (1); w(x) = p / (1 - p) * n_s / n_t estimates p_t / p_s"""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = CalibratedClassifierCV(HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, l2_regularization=1.0, random_state=seed),
                                 method='sigmoid', cv=3)
    clf.fit(np.r_[Xs, Xt], np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]); prior = len(Xs) / len(Xt)
    def w(X, cap=30.0):
        p = np.clip(clf.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6); r = p / (1 - p) * prior; r = r / r.mean()
        return np.minimum(r, cap) / np.minimum(r, cap).mean()
    return w


def ess(w):
    return float(w.sum() ** 2 / (w ** 2).sum())


def wmean_boot(D, w, idx):
    """D (J, m) per-pair values, w (J,) weights, idx (B, J) bootstrap rows -> (m,) weighted mean and (B, m) bootstrap means"""
    d = (w[:, None] * D).sum(0) / w.sum(); W = w[idx]; bs = np.stack([(W * D[idx, i]).sum(1) / W.sum(1) for i in range(D.shape[1])], 1)
    return d, bs


def wquant(x, w, q):
    o = np.argsort(x); c = np.cumsum(w[o]) / w.sum(); return float(x[o][min(np.searchsorted(c, q), len(x) - 1)])


def step_stats(a):
    from scipy.stats import norm
    from sklearn.metrics import roc_auc_score
    from interp import h60_top_hadronisation as H60
    Z = np.load(RES / 'h66_scores.npz'); names = sorted({k.split('|')[1] for k in Z.files if k.startswith('y|')})
    S = {s: {m: Z[f'{s}|{m}'].astype(np.float64) for m in NETS + PERMS if f'{s}|{m}' in Z.files} for s in names}
    O = {s: {k.split('|')[2]: Z[k].astype(np.float64) for k in Z.files if k.startswith(f'obs|{s}|')} for s in names}
    Y = {s: Z[f'y|{s}'].astype(int) for s in names}; K = {s: Z[f'key|{s}'] for s in names}
    for s in names:
        for m in SF: O[s][m] = S[s][m]
    zp = lambda d, se: float(2 * norm.sf(abs(d) / se)) if se > 0 else 1.0
    rng = np.random.default_rng(6617); B = a.boot
    R = dict(node='H66', accuracy={}, weights={}, c2st={}, paired={}, alpha={}, tl_edits={}, data_lr={}, tests=[], decomp={}, wp={})
    # ---- unedited accuracy on every unedited set (unweighted)
    for s in ['T1', 'T2', 'E_base', 'A_V0', 'B_V0']:
        R['accuracy'][s] = {m: dict(acc=float(np.mean((S[s][m] > 0) == (Y[s] == 1))), auc=float(roc_auc_score(Y[s], S[s][m]))) for m in S[s]}
    # ---- density-ratio weights, cross-fitted: fit on half h vs T1, apply to the other half; C2ST on the other half vs T2
    feat = lambda s, cls, fs: np.stack([O[s][k][Y[s] == cls] for k in fs], 1)
    W = {}; other = {'A': 'B', 'B': 'A'}
    for h in 'AB':
        for cls in (0, 1):
            wf = fit_ratio(feat(f'{h}_V0', cls, WF), feat('T1', cls, WF), seed=cls)
            ho = other[h]; W[(ho, cls)] = wf(feat(f'{ho}_V0', cls, WF))
            W[(h, cls, 'self')] = wf(feat(f'{h}_V0', cls, WF))
    cn_ = {0: 'qcd', 1: 'top'}
    for h in 'AB':
        for cls in (0, 1):
            w = W[(h, cls)]; R['weights'][f'{h}_{cn_[cls]}'] = dict(n=len(w), ess=ess(w), ess_frac=ess(w) / len(w), max=float(w.max()),
                                                                   q99=float(np.quantile(w, 0.99)), frac_at_cap=float(np.mean(w >= w.max() - 1e-9)))
    FS = {'main (weighting features)': WF, 'unseen substructure': XF + ['m', 'tau32', 'mW'], 'network scores': SF, 'N only': ['N'],
          'all': WF + XF + SF}
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    for h in 'AB':
        for cls in (0, 1):
            Xt = {f: feat('T2', cls, fs) for f, fs in FS.items()}
            for al in alphas:
                w = W[(h, cls)] ** al; w = w / w.mean()
                for f, fs in FS.items():
                    if al not in (0.0, 1.0) and f != 'main (weighting features)': continue
                    R['c2st'][f'{h}|{cn_[cls]}|{f}|a{al}'] = c2st(feat(f'{h}_V0', cls, fs), Xt[f], w, seed=cls)
            R['c2st'][f'null T1 vs T2|{cn_[cls]}'] = c2st(feat('T1', cls, WF), Xt['main (weighting features)'], None, seed=cls)
        log('C2ST', h, {k: round(v, 3) for k, v in R['c2st'].items() if k.startswith(h + '|') and ('a1.0' in k or 'a0.0' in k)})
    log('C2ST nulls', {k: round(v, 3) for k, v in R['c2st'].items() if k.startswith('null')})
    # ---- feature means: TL vs V0 unweighted vs V0 weighted
    R['means'] = {}
    for cls in (0, 1):
        R['means'][cn_[cls]] = {}
        for k in ['N', 'm', 'tau32', 'mW', 'pTD', 'f1', 'girth', 'n_lt1', 'ParT_top']:
            x = np.concatenate([O[f'{h}_V0'][k][Y[f'{h}_V0'] == cls] for h in 'AB']); w = np.concatenate([W[(h, cls)] for h in 'AB'])
            R['means'][cn_[cls]][k] = dict(TL=float(O['T2'][k][Y['T2'] == cls].mean()), V0=float(x.mean()), V0_weighted=float((w * x).sum() / w.sum()))
    # ---- weighted accuracy on the matched V0 sample (both halves, cross-fitted weights)
    wall = {h: np.zeros(len(Y[f'{h}_V0'])) for h in 'AB'}
    for h in 'AB':
        for cls in (0, 1): wall[h][Y[f'{h}_V0'] == cls] = W[(h, cls)]
    yv = np.concatenate([Y[f'{h}_V0'] for h in 'AB']); wv = np.concatenate([wall[h] for h in 'AB'])
    R['accuracy']['V0_matched_weighted'] = {}
    for m in S['A_V0']:
        s = np.concatenate([S[f'{h}_V0'][m] for h in 'AB'])
        R['accuracy']['V0_matched_weighted'][m] = dict(acc=float(np.sum(wv * ((s > 0) == (yv == 1))) / wv.sum()), auc=float(roc_auc_score(yv, s, sample_weight=wv)))
    # ---- paired shifts on the matched set: Hz+ - V0, cnt - V0, rest = Hz+ - cnt
    models = [m for m in NETS + PERMS if m in S['A_V0']]; mi = {m: i for i, m in enumerate(models)}
    PR = {}
    for h in 'AB':
        p1 = {int(k): j for j, k in enumerate(K[f'{h}_Hzp'])}; pc = {int(k): j for j, k in enumerate(K[f'{h}_cnt'])}
        PR[h] = np.array([(j, p1[int(k)], pc[int(k)]) for j, k in enumerate(K[f'{h}_V0']) if int(k) in p1 and int(k) in pc])
    obsk = ['N', 'm', 'tau32', 'mW', 'pTD', 'f1', 'girth']
    for cls in (0, 1):
        cn = cn_[cls]; Ds = {v: [] for v in ('Hzp', 'cnt', 'rest')}; Do = {v: [] for v in ('Hzp', 'cnt', 'rest')}; ww = []; c0, c1h, c1c = [], [], []
        for h in 'AB':
            pr = PR[h]; sel = Y[f'{h}_V0'][pr[:, 0]] == cls; pr = pr[sel]
            pos = {j: i for i, j in enumerate(np.flatnonzero(Y[f'{h}_V0'] == cls))}; ww.append(W[(h, cls)][[pos[j] for j in pr[:, 0]]])
            s0 = np.stack([S[f'{h}_V0'][m][pr[:, 0]] for m in models], 1); s1 = np.stack([S[f'{h}_Hzp'][m][pr[:, 1]] for m in models], 1)
            s2 = np.stack([S[f'{h}_cnt'][m][pr[:, 2]] for m in models], 1)
            o0 = np.stack([O[f'{h}_V0'][k][pr[:, 0]] for k in obsk], 1); o1 = np.stack([O[f'{h}_Hzp'][k][pr[:, 1]] for k in obsk], 1)
            o2 = np.stack([O[f'{h}_cnt'][k][pr[:, 2]] for k in obsk], 1)
            Ds['Hzp'].append(s1 - s0); Ds['cnt'].append(s2 - s0); Ds['rest'].append(s1 - s2)
            Do['Hzp'].append(o1 - o0); Do['cnt'].append(o2 - o0); Do['rest'].append(o1 - o2); c0.append(s0); c1h.append(s1); c1c.append(s2)
        w = np.concatenate(ww); J = len(w); idx = rng.integers(0, J, (B, J)); s0 = np.concatenate(c0); s1 = {'Hzp': np.concatenate(c1h), 'cnt': np.concatenate(c1c)}
        R['paired'][cn] = {}
        for v in ('Hzp', 'cnt', 'rest'):
            D = np.concatenate(Ds[v]); Dob = np.concatenate(Do[v]); r = dict(n_pairs=J, ess=ess(w), obs={}, models={}, unweighted={})
            dob, bob = wmean_boot(Dob, w, idx[:400])
            for i, k in enumerate(obsk): r['obs'][k] = dict(d=float(dob[i]), ci=np.percentile(bob[:, i], [2.5, 97.5]).tolist(), d_unw=float(Dob[:, i].mean()))
            d, bs = wmean_boot(D, w, idx); du = D.mean(0)
            for m in models:
                i = mi[m]; rr = dict(d=float(d[i]), ci=np.percentile(bs[:, i], [2.5, 97.5]).tolist(), p0=zp(d[i], bs[:, i].std()), d_unweighted=float(du[i]),
                                     per_object=float(d[i] / dob[0]) if abs(dob[0]) > 0.3 else None)
                if m != 'EFN_top' and m in NETS:
                    e = mi['EFN_top']; dd = d[i] - d[e]; bdd = bs[:, i] - bs[:, e]
                    rr['minus_EFN'] = dict(d=float(dd), ci=np.percentile(bdd, [2.5, 97.5]).tolist(), p=zp(dd, bdd.std()))
                    if m in TESTNETS: R['tests'].append(dict(arm=f'matched {v}', cls=cn, model=m, d=float(dd), p=rr['minus_EFN']['p']))
                r['models'][m] = rr
            if v == 'rest':     # H0: dS(Hz+) = dS(cnt), i.e. weighted mean of (Hz+ - cnt) = 0; sign-flip permutation p (2000)
                prng = np.random.default_rng(6619); eps = prng.choice([-1.0, 1.0], (2000, J)); R.setdefault('hzp_vs_cnt', [])
                for m in NETS:
                    x = w * D[:, mi[m]]; t0 = abs(x.sum()); tp = np.abs(eps @ x)
                    pp = float((1 + (tp >= t0).sum()) / 2001); r['models'][m]['p_perm'] = pp
                    R['hzp_vs_cnt'].append(dict(cls=cn, model=m, d=float(d[mi[m]]), ci=r['models'][m]['ci'], p_perm=pp))
            pv = [r['models'][p]['d'] for p in PERMS if p in r['models']]
            r['perm_summary'] = dict(mean=float(np.mean(pv)), min=float(np.min(pv)), max=float(np.max(pv))) if pv else None
            R['paired'][cn][v] = r
            log(f'matched {v} {cn}: pairs {J} ESS {ess(w):.0f} dN {dob[0]:+.2f} dm {dob[1]:+.2f} dtau32 {dob[2]:+.4f} dmW {dob[3]:+.2f} | '
                + ' '.join(f"{m} {r['models'][m]['d']:+.3f} [{r['models'][m]['ci'][0]:+.3f},{r['models'][m]['ci'][1]:+.3f}] (unw {r['models'][m]['d_unweighted']:+.3f})" for m in NETS))
        # working point: weighted median of matched V0 top scores; weighted and unweighted McNemar for Hz+ and cnt
        if cls == 1:
            R['wp'] = {}
            for v in ('Hzp', 'cnt'):
                R['wp'][v] = {}
                for m in models:
                    th = wquant(s0[:, mi[m]], w, 0.5); a0 = s0[:, mi[m]] > th; a1 = s1[v][:, mi[m]] > th
                    g, l, p = H60._mcnemar(a0, a1); dg = (w * (~a0 & a1)) - (w * (a0 & ~a1))
                    bd = np.array([dg[ix].sum() / w[ix].sum() for ix in idx[:1000]])
                    R['wp'][v][m] = dict(eff_v0=float((w * a0).sum() / w.sum()), eff_var=float((w * a1).sum() / w.sum()), gained=g, lost=l, p_mcnemar=p,
                                         d_eff_w=float(dg.sum() / w.sum()), d_eff_w_ci=np.percentile(bd, [2.5, 97.5]).tolist(), p_w=zp(dg.sum() / w.sum(), bd.std()))
        # tempered weights: shift vs alpha (and the C2ST at that alpha, averaged over halves)
        R['alpha'][cn] = []
        D = np.concatenate(Ds['Hzp']); dN = np.concatenate(Do['Hzp'])[:, 0]
        for al in alphas:
            wa = w ** al; d, bs = wmean_boot(D, wa, idx[:500])
            R['alpha'][cn].append(dict(alpha=al, c2st=float(np.mean([R['c2st'][f'{h}|{cn}|main (weighting features)|a{al}'] for h in 'AB'])), ess=ess(wa),
                                       dN=float((wa * dN).sum() / wa.sum()),
                                       models={m: dict(d=float(d[mi[m]]), ci=np.percentile(bs[:, mi[m]], [2.5, 97.5]).tolist()) for m in NETS}))
    # decomposition table (top jets): Hz+ = count part + rest
    for m in models:
        r = R['paired']['top']
        R['decomp'][m] = dict(hzp=r['Hzp']['models'][m]['d'], count=r['cnt']['models'][m]['d'], rest=r['rest']['models'][m]['d'],
                              count_ci=r['cnt']['models'][m]['ci'], rest_ci=r['rest']['models'][m]['ci'])
    # ---- edits on real Top Landscape test jets
    Eb = 'E_base'
    for cls in (0, 1):
        cn = cn_[cls]; sel = Y[Eb] == cls; J = int(sel.sum()); idx = rng.integers(0, J, (B, J)); R['tl_edits'][cn] = {}
        for v in [f'E_sr{k}' for k in KS] + ['E_jit8', 'E_ss8']:
            D = np.stack([S[v][m][sel] - S[Eb][m][sel] for m in models], 1); dN = (O[v]['N'] - O[Eb]['N'])[sel]
            dm = (O[v]['m'] - O[Eb]['m'])[sel]; dt3 = (O[v]['tau32'] - O[Eb]['tau32'])[sel]; dW = (O[v]['mW'] - O[Eb]['mW'])[sel]
            d = D.mean(0); bs = np.stack([D[ix].mean(0) for ix in idx]); r = dict(n=J, dN=float(dN.mean()), dm=float(dm.mean()), dm_maxabs=float(np.abs(dm).max()),
                                                                           dtau32=float(dt3.mean()), dtau32_maxabs=float(np.abs(dt3).max()), dmW_maxabs=float(np.abs(dW).max()), models={})
            for m in models:
                i = mi[m]; rr = dict(d=float(d[i]), ci=np.percentile(bs[:, i], [2.5, 97.5]).tolist(), p0=zp(d[i], bs[:, i].std()),
                                     per_object=float(d[i] / dN.mean()) if dN.mean() > 0.5 else None,
                                     per_object_ci=(np.percentile(bs[:, i], [2.5, 97.5]) / dN.mean()).tolist() if dN.mean() > 0.5 else None)
                if m in NETS and m != 'EFN_top':
                    e = mi['EFN_top']; dd = d[i] - d[e]; bdd = bs[:, i] - bs[:, e]
                    rr['minus_EFN'] = dict(d=float(dd), ci=np.percentile(bdd, [2.5, 97.5]).tolist(), p=zp(dd, bdd.std()))
                    if m in TESTNETS and v != 'E_jit8': R['tests'].append(dict(arm=f'TL {v}', cls=cn, model=m, d=float(dd), p=rr['minus_EFN']['p']))
                # McNemar at the TL 50 % top-efficiency threshold (median of unedited top scores)
                th = np.median(S[Eb][m][Y[Eb] == 1]); a0 = S[Eb][m][sel] > th; a1 = S[v][m][sel] > th; g, l, p = H60._mcnemar(a0, a1)
                rr.update(eff0=float(a0.mean()), eff1=float(a1.mean()), gained=g, lost=l, p_mcnemar=p)
                r['models'][m] = rr
            pv = [r['models'][p]['d'] for p in PERMS if p in r['models']]
            r['perm_summary'] = dict(mean=float(np.mean(pv)), min=float(np.min(pv)), max=float(np.max(pv))) if pv else None
            R['tl_edits'][cn][v] = r
            log(f'TL {v} {cn}: dN {r["dN"]:.2f} max|dm| {r["dm_maxabs"]:.4f} max|dtau32| {r["dtau32_maxabs"]:.5f} | '
                + ' '.join(f"{m} {r['models'][m]['d']:+.3f}" for m in NETS) + f" | perms {r['perm_summary']}")
        # dose slope through the origin over k = 2, 4, 8 (per-jet), bootstrap over jets
        Dk = {m: np.stack([(S[f'E_sr{k}'][m] - S[Eb][m])[sel] for k in KS], 1) for m in models}
        Nk = np.stack([(O[f'E_sr{k}']['N'] - O[Eb]['N'])[sel] for k in KS], 1)
        R['tl_edits'][cn]['slope'] = {}
        for m in models:
            f = lambda ix: float((Dk[m][ix] * Nk[ix]).sum() / (Nk[ix] ** 2).sum())
            bsl = np.array([f(ix) for ix in idx[:1000]])
            R['tl_edits'][cn]['slope'][m] = dict(b=f(np.arange(J)), ci=np.percentile(bsl, [2.5, 97.5]).tolist())
    # ---- Holm across models x arms
    adj = H60._holm([t['p'] for t in R['tests']])
    for t, hh in zip(R['tests'], adj): t['holm'] = float(hh)
    R['holm'] = dict(family='network minus EFN shift, bootstrap-z p; matched Hz+/cnt/rest x 2 classes and TL split_rand k=2,4,8 / split_soft k=8 x 2 classes, x 6 networks',
                     n_tests=len(adj), n_pass_005=int((adj < 0.05).sum()))
    if R.get('hzp_vs_cnt'):
        adj2 = H60._holm([t['p_perm'] for t in R['hzp_vs_cnt']])
        for t, hh in zip(R['hzp_vs_cnt'], adj2): t['holm'] = float(hh)
        log('Hz+ vs cnt (rest) permutation tests', [(t['cls'], t['model'], round(t['d'], 3), t['holm']) for t in R['hzp_vs_cnt']])
    # ---- the data's conditional count ratio on Top Landscape train jets
    R['data_lr'] = data_lr(B=a.boot_lr)
    (RES / 'h66_run.json').write_text(json.dumps(R, indent=1, default=float)); log('wrote h66_run.json')
    fig(R)


def data_lr(B=100):
    """d log-odds(top) / dN at fixed substructure on Top Landscape train jets.
    (1) unconditional logistic regression label ~ N;  (2) fixed-effect (cell) logistic: label ~ cell(m, tau32, mW) + beta N, cells =
    10 m-quantile x 8 tau32-quantile x 8 mW-quantile bins (640 cells), almost unpenalised (C = 1e4); jets in single-class cells
    contribute nothing to beta; also restricted to the top-mass window 140 < m < 200 GeV;  (3) GBDT on (m, tau32, mW, N), 2-fold
    cross-fitted; average finite difference [f(N + 5) - f(N - 5)] / 10 over held-out top and QCD jets; (4) the same with
    (m, tau32, mW, tau21, mSD, pT) as conditioning set.  CIs: bootstrap over jets (B refits for 1-2; jet resampling of the held-out
    finite differences for 3-4)."""
    import scipy.sparse as sp
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    T = np.load(RES / 'h66_tltrain.npz'); y = T['y'].astype(int); N = T['N'].astype(float); rng = np.random.default_rng(6618)
    out = dict(n=int(len(y)), N_mean=dict(top=float(N[y == 1].mean()), qcd=float(N[y == 0].mean())))
    def lr1(ix):
        return float(LogisticRegression(C=1e4, max_iter=1000).fit(N[ix, None] / 10, y[ix]).coef_[0, 0] / 10)
    J = len(y); bix = [rng.integers(0, J, J) for _ in range(B)]
    out['uncond'] = dict(b=lr1(np.arange(J)), ci=np.percentile([lr1(ix) for ix in bix[:B]], [2.5, 97.5]).tolist())
    q = lambda x, k: np.clip(np.searchsorted(np.quantile(x, np.linspace(0, 1, k + 1)[1:-1]), x, side='right'), 0, k - 1)
    cell = (q(T['m'], 10) * 8 + q(T['tau32'], 8)) * 8 + q(T['mW'], 8)
    def fe(ix):
        c = cell[ix]; u, inv = np.unique(c, return_inverse=True)
        X = sp.hstack([sp.csr_matrix((np.ones(len(ix)), (np.arange(len(ix)), inv)), shape=(len(ix), len(u))), sp.csr_matrix(((N[ix] - 50) / 10)[:, None])]).tocsr()
        m = LogisticRegression(C=1e4, fit_intercept=False, max_iter=3000).fit(X, y[ix]); return float(m.coef_[0, -1] / 10)
    for tag, base in (('fe_all', np.arange(J)), ('fe_mwin', np.flatnonzero((T['m'] > 140) & (T['m'] < 200)))):
        b = fe(base); bs = [fe(base[rng.integers(0, len(base), len(base))]) for _ in range(B)]
        out[tag] = dict(b=b, ci=np.percentile(bs, [2.5, 97.5]).tolist(), n=int(len(base)))
        log('data LR', tag, out[tag])
    log('data LR uncond', out['uncond'])
    for tag, cols in (('gb_3', ['m', 'tau32', 'mW']), ('gb_6', ['m', 'tau32', 'mW', 'tau21', 'mSD', 'jpt'])):
        Xc = np.stack([T[c].astype(float) for c in cols], 1); fd = np.zeros(J)
        for tr, te in StratifiedKFold(2, shuffle=True, random_state=0).split(Xc, y):
            clf = _hgb(0, 300).fit(np.c_[Xc[tr], N[tr]], y[tr])
            fd[te] = (clf.decision_function(np.c_[Xc[te], N[te] + 5]) - clf.decision_function(np.c_[Xc[te], np.maximum(N[te] - 5, 1)])) / 10
        r = {}
        for cls, cn in ((1, 'top'), (0, 'qcd')):
            x = fd[y == cls]; bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(1000)]
            r[cn] = dict(b=float(x.mean()), ci=np.percentile(bs, [2.5, 97.5]).tolist())
        mw = (T['m'] > 140) & (T['m'] < 200) & (y == 1); r['top_mwin'] = dict(b=float(fd[mw].mean()), n=int(mw.sum()))
        out[tag] = r; log('data LR', tag, r)
    # per-m-bin fixed-effect slope (tau32 x mW cells inside 20-GeV m bins), for the figure
    out['by_m'] = []
    for lo in range(40, 260, 20):
        ix = np.flatnonzero((T['m'] >= lo) & (T['m'] < lo + 20))
        if len(ix) < 2000 or len(np.unique(y[ix])) < 2: continue
        c2 = q(T['tau32'][ix], 8) * 8 + q(T['mW'][ix], 8); u, inv = np.unique(c2, return_inverse=True)
        X = sp.hstack([sp.csr_matrix((np.ones(len(ix)), (np.arange(len(ix)), inv)), shape=(len(ix), len(u))), sp.csr_matrix(((N[ix] - 50) / 10)[:, None])]).tocsr()
        m = LogisticRegression(C=1e4, fit_intercept=False, max_iter=3000).fit(X, y[ix])
        out['by_m'].append(dict(m_lo=lo, n=int(len(ix)), top_frac=float(y[ix].mean()), b=float(m.coef_[0, -1] / 10)))
    return out


def step_sym(a):
    """Pair-symmetric weights and per-half consistency.
    The stats step weights each pair by w(V0 jet).  Because V0 and Hz+ are two noisy hadronisations of one parton event, selecting
    on the V0 member's N, m, tau32, mW selects V0 fluctuations, and Hz+ - V0 then contains a regression-to-the-mean shift (the weighted
    Hz+ - V0 has dN 7.6 instead of 4.5 and dm -1.2 GeV).  Here each pair gets w_sym = sqrt(w(V0 features) * w(Hz+ features)) (same
    cross-fitted ratio model, fitted on V0 vs T1), which depends on the pair symmetrically.  Reported: C2ST of V0 and of Hz+ under
    w_sym vs T2, paired shifts (Hz+, cnt, rest) under unweighted / w(V0) / w_sym, per half and pooled, per object, sign-flip p."""
    from interp import h60_top_hadronisation as H60
    Z = np.load(RES / 'h66_scores.npz'); names = sorted({k.split('|')[1] for k in Z.files if k.startswith('y|')})
    S = {s: {m: Z[f'{s}|{m}'].astype(np.float64) for m in NETS if f'{s}|{m}' in Z.files} for s in names}
    O = {s: {k.split('|')[2]: Z[k].astype(np.float64) for k in Z.files if k.startswith(f'obs|{s}|')} for s in names}
    Y = {s: Z[f'y|{s}'].astype(int) for s in names}; K = {s: Z[f'key|{s}'] for s in names}
    F = lambda s, rows: np.stack([O[s][k][rows] for k in WF], 1)
    rng = np.random.default_rng(6620); B = a.boot; other = {'A': 'B', 'B': 'A'}; cn_ = {0: 'qcd', 1: 'top'}
    obsk = ['N', 'm', 'tau32', 'mW', 'pTD']
    out = dict(node='H66 sym', c2st={}, shifts={}, ess={})
    for cls in (0, 1):
        cn = cn_[cls]; pool = {}
        for h in 'AB':
            ho = other[h]; r0 = np.flatnonzero(Y[f'{ho}_V0'] == cls)
            wf = fit_ratio(F(f'{ho}_V0', r0), F('T1', np.flatnonzero(Y['T1'] == cls)), seed=cls)      # same fit as step_stats (fit half = ho)
            p1 = {int(k): j for j, k in enumerate(K[f'{h}_Hzp'])}; pc = {int(k): j for j, k in enumerate(K[f'{h}_cnt'])}
            pr = np.array([(j, p1[int(k)], pc[int(k)]) for j, k in enumerate(K[f'{h}_V0'])
                           if int(k) in p1 and int(k) in pc and Y[f'{h}_V0'][j] == cls])
            w0 = wf(F(f'{h}_V0', pr[:, 0])); w1 = wf(F(f'{h}_Hzp', pr[:, 1])); ws = np.sqrt(w0 * w1); ws /= ws.mean()
            Xt = F('T2', np.flatnonzero(Y['T2'] == cls))
            out['c2st'][f'{h}|{cn}'] = dict(V0_w0=c2st(F(f'{h}_V0', pr[:, 0]), Xt, w0, seed=cls), V0_wsym=c2st(F(f'{h}_V0', pr[:, 0]), Xt, ws, seed=cls),
                                           Hzp_wsym=c2st(F(f'{h}_Hzp', pr[:, 1]), Xt, ws, seed=cls), Hzp_unw=c2st(F(f'{h}_Hzp', pr[:, 1]), Xt, None, seed=cls),
                                           cnt_wsym=c2st(F(f'{h}_cnt', pr[:, 2]), Xt, ws, seed=cls))
            log('C2ST', h, cn, {k: round(v, 3) for k, v in out['c2st'][f'{h}|{cn}'].items()})
            s0 = np.stack([S[f'{h}_V0'][m][pr[:, 0]] for m in NETS], 1); s1 = np.stack([S[f'{h}_Hzp'][m][pr[:, 1]] for m in NETS], 1)
            s2 = np.stack([S[f'{h}_cnt'][m][pr[:, 2]] for m in NETS], 1)
            o0 = np.stack([O[f'{h}_V0'][k][pr[:, 0]] for k in obsk], 1); o1 = np.stack([O[f'{h}_Hzp'][k][pr[:, 1]] for k in obsk], 1)
            o2 = np.stack([O[f'{h}_cnt'][k][pr[:, 2]] for k in obsk], 1)
            pool[h] = dict(W={'unw': np.ones(len(pr)), 'w_v0': w0, 'w_sym': ws}, D={'Hzp': s1 - s0, 'cnt': s2 - s0, 'rest': s1 - s2},
                           Do={'Hzp': o1 - o0, 'cnt': o2 - o0, 'rest': o1 - o2})
        pool['pooled'] = {k: {kk: np.concatenate([pool[h][k][kk] for h in 'AB']) for kk in pool['A'][k]} for k in ('W', 'D', 'Do')}
        out['shifts'][cn] = {}
        for h in ('A', 'B', 'pooled'):
            P = pool[h]; J = len(P['W']['unw']); idx = rng.integers(0, J, (B, J)); out['shifts'][cn][h] = {}
            for wn, w in P['W'].items():
                out['ess'][f'{h}|{cn}|{wn}'] = ess(w); rr = {}
                for v in ('Hzp', 'cnt', 'rest'):
                    d, bs = wmean_boot(P['D'][v], w, idx); dob, bob = wmean_boot(P['Do'][v], w, idx[:300])
                    e = dict(n=J, obs={k: dict(d=float(dob[i]), ci=np.percentile(bob[:, i], [2.5, 97.5]).tolist()) for i, k in enumerate(obsk)}, models={})
                    for i, m in enumerate(NETS):
                        e['models'][m] = dict(d=float(d[i]), ci=np.percentile(bs[:, i], [2.5, 97.5]).tolist(),
                                              per_object=float(d[i] / dob[0]) if abs(dob[0]) > 0.3 else None,
                                              per_object_ci=np.percentile(bs[:, i] / bob[:, 0].mean(), [2.5, 97.5]).tolist() if abs(dob[0]) > 0.3 else None)
                        if v == 'rest' and h == 'pooled':
                            x = w * P['D'][v][:, i]; eps = np.random.default_rng(6621).choice([-1.0, 1.0], (2000, J))
                            e['models'][m]['p_perm'] = float((1 + (np.abs(eps @ x) >= abs(x.sum())).sum()) / 2001)
                    rr[v] = e
                out['shifts'][cn][h][wn] = rr
                log(f'{cn} {h} {wn}: ESS {ess(w):.0f} | ' + ' | '.join(
                    f"{v} dN {rr[v]['obs']['N']['d']:+.2f} dm {rr[v]['obs']['m']['d']:+.2f} dt32 {rr[v]['obs']['tau32']['d']:+.4f} "
                    + ' '.join(f"{m} {rr[v]['models'][m]['d']:+.3f}" for m in NETS) for v in ('Hzp', 'cnt', 'rest')))
    tests = [(cn, m) for cn in ('top', 'qcd') for m in NETS]
    adj = H60._holm([out['shifts'][cn]['pooled']['w_sym']['rest']['models'][m]['p_perm'] for cn, m in tests])
    out['holm_rest_wsym'] = [dict(cls=cn, model=m, holm=float(x)) for (cn, m), x in zip(tests, adj)]
    (RES / 'h66_sym.json').write_text(json.dumps(out, indent=1, default=float)); log('wrote h66_sym.json')
    fig_sym(out)


def fig_sym(out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    show = [('ParT_top', 'ParT'), ('PN_top', 'ParticleNet'), ('LNet_top', 'LorentzNet'), ('PFN_top', 'PFN'), ('ParT_kin', 'ParT_kin\n(JetClass)'),
            ('PN_kin', 'PN_kin\n(JetClass)'), ('EFN_top', 'EFN\n(IRC-safe)')]
    W = [('unw', 'unweighted (C2ST ≈ 0.80)', '#9e9e9e'), ('w_v0', 'weights from the V0 jet', '#ff7f0e'), ('w_sym', 'pair-symmetric weights', '#1f77b4')]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.0), sharey=False)
    for ax, v, title in ((axs[0], 'Hzp', 'Hz+ minus V0 (hadronisation change)'), (axs[1], 'cnt', 'count arm minus V0 (exact collinear splits)')):
        x = np.arange(len(show))
        for j, (wn, lab, c) in enumerate(W):
            e = out['shifts']['top']['pooled'][wn][v]
            d = np.array([e['models'][m]['d'] for m, _ in show]); ci = np.array([e['models'][m]['ci'] for m, _ in show])
            ax.errorbar(x + (j - 1) * 0.22, d, yerr=[d - ci[:, 0], ci[:, 1] - d], fmt='o', color=c, ms=5, capsize=2,
                        label=f"{lab}; ΔN = {e['obs']['N']['d']:+.1f}")
        ax.axhline(0, color='k', lw=0.8); ax.set_xticks(x); ax.set_xticklabels([s for _, s in show], fontsize=8.5)
        ax.set_title(f'Top jets: {title}', fontsize=10); ax.legend(fontsize=8, loc='lower right')
    axs[0].set_ylabel('mean change in top log-odds per jet pair')
    fig.tight_layout(); fig.savefig(RES / 'h66_sym.png', dpi=130); log('wrote h66_sym.png')


def fig(R):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    show = [('ParT_top', 'ParT (Top)', '#1f77b4'), ('PN_top', 'ParticleNet (Top)', '#ff7f0e'), ('LNet_top', 'LorentzNet (Top)', '#2ca02c'),
            ('PFN_top', 'PFN (Top)', '#9467bd'), ('ParT_kin', 'ParT_kin (JetClass)', '#8c564b'), ('EFN_top', 'EFN (IRC-safe)', '#7f7f7f')]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.4))
    ax = axs[0]
    for k, (m, lab, c) in enumerate(show):
        A = R['alpha']['top']; x = np.array([r['c2st'] for r in A]) + 0.0015 * (k - 2.5)
        d = np.array([r['models'][m]['d'] for r in A]); ci = np.array([r['models'][m]['ci'] for r in A])
        ax.errorbar(x, d, yerr=[d - ci[:, 0], ci[:, 1] - d], fmt='o-', color=c, ms=4, lw=1.2, capsize=2, label=lab)
    ax.axhline(0, color='k', lw=0.8); ax.axvline(0.6, color='0.5', ls='--', lw=0.8); ax.text(0.601, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 0.02, 'target 0.6', fontsize=8, color='0.4')
    ax.set_xlabel('C2ST AUC, Delphes V0 top jets vs Top Landscape test (0.5 = indistinguishable)')
    ax.set_ylabel('mean change in top log-odds, Hz+ minus V0')
    ax.set_title('Top jets: Hz+ (+7 % objects at fixed m, τ32, mW)\nas the generated jets are reweighted toward Top Landscape', fontsize=10)
    ax.legend(fontsize=8, loc='lower left'); ax.invert_xaxis()
    ax = axs[1]; T = R['tl_edits']['top']['slope']; DL = R['data_lr']
    rows = [(lab, T[m]['b'], T[m]['ci'], c) for m, lab, c in show]
    rows += [('data: LR(N) unconditional', DL['uncond']['b'], DL['uncond']['ci'], '#d62728'),
             ('data: LR(N | m, τ32, mW cells)', DL['fe_all']['b'], DL['fe_all']['ci'], '#d62728'),
             ('data: same, 140 < m < 200 GeV', DL['fe_mwin']['b'], DL['fe_mwin']['ci'], '#d62728'),
             ('data: GBDT(m, τ32, mW, N), top jets', DL['gb_3']['top']['b'], DL['gb_3']['top']['ci'], '#d62728')]
    yy = np.arange(len(rows))
    for i, (lab, b, ci, c) in enumerate(rows):
        ax.errorbar([b], [i], xerr=[[b - ci[0]], [ci[1] - b]], fmt='s' if lab.startswith('data') else 'o', color=c, ms=5, capsize=2)
    ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows], fontsize=8.5); ax.invert_yaxis(); ax.axvline(0, color='k', lw=0.8)
    ax.axhline(len(show) - 0.5, color='0.6', lw=0.6, ls='--')
    ax.set_xlabel('change in top log-odds per added object')
    ax.set_title('Real Top Landscape top jets: network response per collinear split\nvs the data\'s own log-odds slope in N (red)', fontsize=10)
    fig.tight_layout(); fig.savefig(RES / 'h66_count_onmanifold.png', dpi=130); log('wrote h66_count_onmanifold.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', required=True, choices=['score', 'tltrain', 'stats', 'fig', 'sym'])
    ap.add_argument('--boot', type=int, default=2000); ap.add_argument('--boot_lr', type=int, default=100)
    ap.add_argument('--ntrain', type=int, default=60000)
    a = ap.parse_args()
    dict(score=step_score, tltrain=step_tltrain, stats=step_stats, sym=step_sym, fig=lambda a: fig(json.loads((RES / 'h66_run.json').read_text())))[a.step](a)
