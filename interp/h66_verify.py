"""H66 verify (loop D): independent checks of the H66 on-manifold count result.

Step 'edits' (GPU, env-part): fresh Top Landscape test rows (rng 7766, 4,000 per class, disjoint from the H66 rng-6611 draw).
  Edits on each jet (pT-sorted rows; massless constituents, as stored):
    sr8    8 random constituents each split into two pT/2 halves 0.01 apart in (eta, phi) (as H66)       N + 8
    sr8w   the same 8 constituents split with halves 0.05 apart (wider; not near-duplicates)           N + 8
    jit8   the same 8 constituents moved by 0.005 (N fixed; control for sr8)
    jit8w  the same 8 constituents moved by 0.025 (N fixed; control for sr8w)
    ss8    the 8 softest constituents split 0.01 apart (as H66)                                         N + 8
    mg8    the 8 nearest-in-dR pairs of the softer half merged (three-momenta added, massless)           N - 8
    add8   8 soft particles of pT 0.5 GeV each added uniformly in a dR < 0.4 disc around the jet axis    N + 8
  Models: TL ParT, ParticleNet, LorentzNet, PFN, EFN; JetClass ParT_kin, PN_kin; six weight-permuted copies of the TL ParT.
Step 'w' (CPU): reweighting robustness on the H66 paired scores (h66_scores.npz).  The weight model is refitted with the roles of
  the Top Landscape rows swapped (fit V0 vs T2, C2ST vs T1), a different HGB seed and settings, cross-fitted over halves A/B.
  Weightings: unweighted, w(V0), w(Hz+), w_sym = sqrt(w(V0) w(Hz+)), w_avg = (w(V0) + w(Hz+)) / 2.  Also the weight-permuted ParT
  band under each weighting (H66 gave it only under w(V0)).
Step 'data' (CPU, conda): the data count slope b_N on fresh Top Landscape train rows (rng 7768, disjoint from rng 6612), with the
  H66 cells (10 m x 8 tau32 x 8 mW quantile bins), finer cells (20 x 10 x 10) and the H66 cells x 4 jet-pT quartiles.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from interp import h66_top_count_onmanifold as H66

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
RES = H66.RES
NETS = H66.NETS
PERMS = H66.PERMS


def add_soft(x, k, rng, pt_s=0.5, R=0.4):
    px, py, pz = (x[:, 0] * np.cos(x[:, 2])).sum(), (x[:, 0] * np.sin(x[:, 2])).sum(), (x[:, 0] * np.sinh(x[:, 1])).sum()
    jeta = np.arcsinh(pz / np.hypot(px, py)); jphi = np.arctan2(py, px)
    r = R * np.sqrt(rng.uniform(0, 1, k)); th = rng.uniform(0, 2 * np.pi, k)
    a = np.c_[np.full(k, pt_s), jeta + r * np.cos(th), jphi + r * np.sin(th), np.zeros(k)]
    y = np.r_[x, a]; return y[np.argsort(-y[:, 0], kind='stable')]


def step_edits(a):
    import torch
    from sklearn.metrics import roc_auc_score
    from interp import data_top as dt
    from interp import h4_top_wmass as H
    from interp import h19_wpair_cross as X
    from interp import h61_top_delphes as H61
    y = np.load(dt.DST / 'test' / 'label.npy'); rng = np.random.default_rng(6611)
    old = np.concatenate([rng.choice(np.flatnonzero(y == c), 20000, replace=False) for c in (0, 1)])
    free = np.ones(len(y), bool); free[old] = False; rng = np.random.default_rng(7766)
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero((y == c) & free), a.per, replace=False) for c in (0, 1)]))
    full = H._full_test(); jn = np.load(dt.DST / 'test' / 'jet.npy')[:, 5].astype(np.int64)
    P = np.asarray(full[idx][..., :4], np.float32); n = np.minimum(jn[idx], P.shape[1]); del full
    P = H._identity(P, n).astype(np.float32); yy = y[idx].astype(int)
    log(f'fresh TL rows {len(idx)} (top {yy.sum()}), overlap with H66 rows {np.isin(idx, old).sum()}')
    base = H66.rows_from_P4(P, n); rng = np.random.default_rng(7767); picks = [H66.pick(x, 8, rng) for x in base]
    E = {'base': base}
    E['sr8'] = [H66.split_at(x, c, t, 0.01) for x, (c, t) in zip(base, picks)]
    E['sr8w'] = [H66.split_at(x, c, t, 0.05) for x, (c, t) in zip(base, picks)]
    E['jit8'] = [H66.jitter_at(x, c, t, 0.01) for x, (c, t) in zip(base, picks)]
    E['jit8w'] = [H66.jitter_at(x, c, t, 0.05) for x, (c, t) in zip(base, picks)]
    E['ss8'] = [H66.split_at(x, np.argsort(x[:, 0], kind='stable')[:min(8, len(x) - 1)], t, 0.01) for x, (_, t) in zip(base, picks)]
    E['mg8'] = [H61.merge_soft(x, 8) for x in base]
    rs = np.random.default_rng(7769); E['add8'] = [add_soft(x, 8, rs) for x in base]
    Lmax = int(n.max()) + 8; order = np.random.default_rng(7770).permutation(len(n))
    M = X.Models(); M.massless = True; perms = []
    for i in range(H66.NPERM):
        net = H.WeaverNet('ParT', X.RES / 'h4_ckpt_part_kin.pt', M.dev); g = torch.Generator().manual_seed(i)
        with torch.no_grad():
            for prm in net.model.parameters():
                fl = prm.data.view(-1); fl.copy_(fl[torch.randperm(fl.numel(), generator=g).to(fl.device)])
        perms.append(net)
    out = dict(y=yy.astype(np.int8), idx=idx)
    for nm, L in E.items():
        t = time.time(); Pe, ne = H66.P4_from_rows(L, Lmax); O = H66.all_obs(Pe, ne)
        for k in ('N', 'm', 'tau32', 'mW', 'jpt'): out[f'obs|{nm}|{k}'] = O[k].astype(np.float32)
        sc = M.score('top', Pe, ne, X.top_base(Pe, ne), order)
        for i, net in enumerate(perms): sc[f'perm{i}'] = net.score(Pe, ne)
        for k, v in sc.items(): out[f'{nm}|{k}'] = np.asarray(v, np.float32)
        log(f'{nm}: <N> {ne.mean():.2f} {time.time() - t:.0f}s | ' + '; '.join(f'{k} acc {np.mean((v > 0) == (yy == 1)):.4f} auc {roc_auc_score(yy, v):.4f}'
                                                                          for k, v in sc.items() if not k.startswith('perm')))
    np.savez_compressed(RES / 'h66_verify_edits.npz', **out); log('wrote h66_verify_edits.npz')
    summarize_edits()


def summarize_edits():
    Z = np.load(RES / 'h66_verify_edits.npz'); y = Z['y'].astype(int); rng = np.random.default_rng(7771)
    models = [m for m in NETS + PERMS if f'base|{m}' in Z.files]; R = dict(node='H66 verify edits', n=int(len(y)), accuracy={}, edits={})
    from sklearn.metrics import roc_auc_score
    for m in models:
        s = Z[f'base|{m}']; R['accuracy'][m] = dict(acc=float(np.mean((s > 0) == (y == 1))), auc=float(roc_auc_score(y, s)))
    for cls, cn in ((1, 'top'), (0, 'qcd')):
        sel = y == cls; J = int(sel.sum()); idx = rng.integers(0, J, (2000, J)); R['edits'][cn] = {}
        for v in ('sr8', 'sr8w', 'jit8', 'jit8w', 'ss8', 'mg8', 'add8'):
            dN = (Z[f'obs|{v}|N'] - Z['obs|base|N'])[sel]; d_o = {k: (Z[f'obs|{v}|{k}'] - Z[f'obs|base|{k}'])[sel].astype(float) for k in ('m', 'tau32', 'mW', 'jpt')}
            r = dict(dN=float(dN.mean()), **{f'd{k}': float(x.mean()) for k, x in d_o.items()}, dm_absmax=float(np.abs(d_o['m']).max()), models={})
            for m in models:
                D = (Z[f'{v}|{m}'] - Z[f'base|{m}'])[sel].astype(float); bs = D[idx].mean(1)
                r['models'][m] = dict(d=float(D.mean()), ci=np.percentile(bs, [2.5, 97.5]).tolist(),
                                      per_object=float(D.mean() / dN.mean()) if abs(dN.mean()) > 0.5 else None)
            pv = [r['models'][p]['d'] for p in PERMS if p in r['models']]; r['perm_band'] = [float(min(pv)), float(max(pv))]
            R['edits'][cn][v] = r
            log(f'{cn} {v}: dN {r["dN"]:+.2f} dm {r["dm"]:+.4f} (max {r["dm_absmax"]:.3f}) dtau32 {r["dtau32"]:+.5f} | '
                + ' '.join(f"{m} {r['models'][m]['d']:+.3f}" for m in NETS) + f' | perm {r["perm_band"]}')
        # net count response = split minus its matched jitter (same particles moved the same distance, N fixed)
        R['edits'][cn]['split_minus_jitter'] = {}
        for sv, jv in (('sr8', 'jit8'), ('sr8w', 'jit8w')):
            e = {}
            for m in models:
                D = ((Z[f'{sv}|{m}'] - Z[f'base|{m}']) - (Z[f'{jv}|{m}'] - Z[f'base|{m}']))[sel].astype(float); bs = D[idx].mean(1)
                e[m] = dict(d=float(D.mean()), ci=np.percentile(bs, [2.5, 97.5]).tolist(), per_object=float(D.mean() / 8))
            R['edits'][cn]['split_minus_jitter'][sv] = e
    (RES / 'h66_verify_edits.json').write_text(json.dumps(R, indent=1)); log('wrote h66_verify_edits.json')


def fit_ratio2(Xs, Xt, seed):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = CalibratedClassifierCV(HistGradientBoostingClassifier(max_iter=400, learning_rate=0.04, max_leaf_nodes=25, l2_regularization=2.0,
                                                                random_state=seed), method='sigmoid', cv=3)
    clf.fit(np.r_[Xs, Xt], np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]); prior = len(Xs) / len(Xt)
    def w(X, cap=30.0):
        p = np.clip(clf.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6); r = p / (1 - p) * prior; r = r / r.mean()
        return np.minimum(r, cap) / np.minimum(r, cap).mean()
    return w


def step_w(a):
    Z = np.load(RES / 'h66_scores.npz'); names = sorted({k.split('|')[1] for k in Z.files if k.startswith('y|')})
    names = [s for s in names if s in ('A_V0', 'A_Hzp', 'A_cnt', 'B_V0', 'B_Hzp', 'B_cnt', 'T1', 'T2')]
    S = {s: {m: Z[f'{s}|{m}'].astype(np.float64) for m in NETS + PERMS if f'{s}|{m}' in Z.files} for s in names}
    O = {s: {k: Z[f'obs|{s}|{k}'].astype(np.float64) for k in H66.WF} for s in names}
    Y = {s: Z[f'y|{s}'].astype(int) for s in names}; K = {s: Z[f'key|{s}'] for s in names}
    F = lambda s, rows: np.stack([O[s][k][rows] for k in H66.WF], 1)
    other = {'A': 'B', 'B': 'A'}; cn_ = {0: 'qcd', 1: 'top'}; rng = np.random.default_rng(7772); B = a.boot
    models = [m for m in NETS + PERMS if m in S['A_V0']]
    out = dict(node='H66 verify weights', note='ratio model refitted on V0 vs T2 (roles swapped), C2ST vs T1; HGB seed 100+cls, max_iter 400, lr 0.04, 25 leaves, l2 2',
               c2st={}, ess={}, shifts={})
    for cls in (0, 1):
        cn = cn_[cls]; pool = {}
        for h in 'AB':
            ho = other[h]
            wf = fit_ratio2(F(f'{ho}_V0', np.flatnonzero(Y[f'{ho}_V0'] == cls)), F('T2', np.flatnonzero(Y['T2'] == cls)), seed=100 + cls)
            p1 = {int(k): j for j, k in enumerate(K[f'{h}_Hzp'])}; pc = {int(k): j for j, k in enumerate(K[f'{h}_cnt'])}
            pr = np.array([(j, p1[int(k)], pc[int(k)]) for j, k in enumerate(K[f'{h}_V0']) if int(k) in p1 and int(k) in pc and Y[f'{h}_V0'][j] == cls])
            w0 = wf(F(f'{h}_V0', pr[:, 0])); w1 = wf(F(f'{h}_Hzp', pr[:, 1]))
            W = {'unw': np.ones(len(pr)), 'w_v0': w0, 'w_hzp': w1, 'w_sym': np.sqrt(w0 * w1), 'w_avg': 0.5 * (w0 + w1)}
            W = {k: v / v.mean() for k, v in W.items()}
            Xt = F('T1', np.flatnonzero(Y['T1'] == cls))
            out['c2st'][f'{h}|{cn}'] = {f'V0|{wn}': H66.c2st(F(f'{h}_V0', pr[:, 0]), Xt, W[wn], seed=cls) for wn in ('unw', 'w_v0', 'w_sym', 'w_avg')}
            out['c2st'][f'{h}|{cn}'].update({f'Hzp|{wn}': H66.c2st(F(f'{h}_Hzp', pr[:, 1]), Xt, W[wn], seed=cls) for wn in ('unw', 'w_hzp', 'w_sym', 'w_avg')})
            log('C2ST', h, cn, {k: round(v, 3) for k, v in out['c2st'][f'{h}|{cn}'].items()})
            s0 = np.stack([S[f'{h}_V0'][m][pr[:, 0]] for m in models], 1); s1 = np.stack([S[f'{h}_Hzp'][m][pr[:, 1]] for m in models], 1)
            s2 = np.stack([S[f'{h}_cnt'][m][pr[:, 2]] for m in models], 1)
            dN = O[f'{h}_Hzp']['N'][pr[:, 1]] - O[f'{h}_V0']['N'][pr[:, 0]]; dm = Z[f'obs|{h}_Hzp|m'][pr[:, 1]] - Z[f'obs|{h}_V0|m'][pr[:, 0]]
            pool[h] = dict(W=W, D={'Hzp': s1 - s0, 'cnt': s2 - s0, 'rest': s1 - s2}, dN=dN, dm=dm.astype(float))
        P = {k: {kk: np.concatenate([pool[h][k][kk] for h in 'AB']) for kk in pool['A'][k]} for k in ('W', 'D')}
        dN = np.concatenate([pool[h]['dN'] for h in 'AB']); dm = np.concatenate([pool[h]['dm'] for h in 'AB'])
        J = len(dN); idx = rng.integers(0, J, (B, J)); out['shifts'][cn] = {}
        for wn, w in P['W'].items():
            out['ess'][f'{cn}|{wn}'] = H66.ess(w); rr = dict(dN=float((w * dN).sum() / w.sum()), dm=float((w * dm).sum() / w.sum()))
            for v in ('Hzp', 'cnt', 'rest'):
                d, bs = H66.wmean_boot(P['D'][v], w, idx); e = {}
                for i, m in enumerate(models): e[m] = dict(d=float(d[i]), ci=np.percentile(bs[:, i], [2.5, 97.5]).tolist())
                pv = [e[p]['d'] for p in PERMS if p in e]; e['perm_band'] = [float(min(pv)), float(max(pv))]
                rr[v] = e
            for m in ('ParT_top', 'PN_top', 'LNet_top', 'PFN_top'):
                rr.setdefault('count_share', {})[m] = rr['cnt'][m]['d'] / rr['Hzp'][m]['d']
            out['shifts'][cn][wn] = rr
            log(f'{cn} {wn}: ESS {H66.ess(w):.0f} dN {rr["dN"]:+.2f} dm {rr["dm"]:+.2f} | Hzp ' + ' '.join(f"{m} {rr['Hzp'][m]['d']:+.3f}" for m in NETS)
                + f" perm {rr['Hzp']['perm_band']} | cnt ParT {rr['cnt']['ParT_top']['d']:+.3f}")
    (RES / 'h66_verify_w.json').write_text(json.dumps(out, indent=1)); log('wrote h66_verify_w.json')


def step_data(a):
    import scipy.sparse as sp
    from sklearn.linear_model import LogisticRegression
    from interp import data_top as dt
    from interp import h4_top_wmass as H
    d = dt.DST / 'train'; y = np.load(d / 'label.npy'); rng = np.random.default_rng(6612)
    old = np.concatenate([rng.choice(np.flatnonzero(y == c), 60000, replace=False) for c in (0, 1)])
    free = np.ones(len(y), bool); free[old] = False; rng = np.random.default_rng(7768)
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero((y == c) & free), a.ntrain, replace=False) for c in (0, 1)]))
    k3m = np.load(d / 'kin3.npy', mmap_mode='r'); K = np.zeros((len(idx), k3m.shape[1], 3), np.float32); Bk = 100_000
    for s in range(0, k3m.shape[0], Bk):
        sel = (idx >= s) & (idx < s + Bk)
        if sel.any(): blk = np.asarray(k3m[s:s + Bk]); K[sel] = blk[idx[sel] - s]
    jet = np.load(d / 'jet.npy')[idx]; log(f'TL train fresh rows {len(idx)}, overlap with H66 rows {np.isin(idx, old).sum()}')
    n = (K[..., 0] > 0).sum(1).astype(np.int64); pt, eta, phi = K[..., 0].astype(np.float64), K[..., 1].astype(np.float64), K[..., 2].astype(np.float64)
    P = (np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)], -1) * (pt > 0)[..., None]).astype(np.float32); del K
    o = H._obs(P, n); del P; log('obs done; m check median |m - jet_mass|', float(np.median(np.abs(o['m'] - jet[:, 4]))))
    yy = y[idx].astype(int); N = jet[:, 5].astype(float); jpt = jet[:, 0].astype(float)
    q = lambda x, k: np.clip(np.searchsorted(np.quantile(x, np.linspace(0, 1, k + 1)[1:-1]), x, side='right'), 0, k - 1)
    cells = {'h66_10x8x8': (q(o['m'], 10) * 8 + q(o['tau32'], 8)) * 8 + q(o['mW'], 8),
             'fine_20x10x10': (q(o['m'], 20) * 10 + q(o['tau32'], 10)) * 10 + q(o['mW'], 10)}
    cells['h66_x_pT4'] = cells['h66_10x8x8'] * 4 + q(jpt, 4)
    def fe(ix, cell):
        c = cell[ix]; u, inv = np.unique(c, return_inverse=True)
        X = sp.hstack([sp.csr_matrix((np.ones(len(ix)), (np.arange(len(ix)), inv)), shape=(len(ix), len(u))), sp.csr_matrix(((N[ix] - 50) / 10)[:, None])]).tocsr()
        return float(LogisticRegression(C=1e4, fit_intercept=False, max_iter=3000).fit(X, yy[ix]).coef_[0, -1] / 10)
    out = dict(node='H66 verify data b_N', n=int(len(idx)), N_mean=dict(top=float(N[yy == 1].mean()), qcd=float(N[yy == 0].mean())), fits={})
    J = len(idx); mw = np.flatnonzero((o['m'] > 140) & (o['m'] < 200))
    for tag, cell in cells.items():
        for sub, base in (('all', np.arange(J)), ('140<m<200', mw)):
            b = fe(base, cell); bs = [fe(base[rng.integers(0, len(base), len(base))], cell) for _ in range(a.boot_lr)]
            out['fits'][f'{tag}|{sub}'] = dict(b=b, ci=np.percentile(bs, [2.5, 97.5]).tolist(), n=int(len(base)))
            log('b_N', tag, sub, out['fits'][f'{tag}|{sub}'])
    # shuffled-N null within cells (H0: no N dependence at fixed cell): permute N inside each H66 cell
    cell = cells['h66_10x8x8']; Nsave = N.copy(); nulls = []
    for r in range(20):
        Np = N.copy()
        for c in np.unique(cell):
            ix = np.flatnonzero(cell == c); Np[ix] = Np[rng.permutation(ix)]
        N = Np; nulls.append(fe(np.arange(J), cell)); N = Nsave
    out['null_within_cell_shuffle'] = dict(mean=float(np.mean(nulls)), min=float(np.min(nulls)), max=float(np.max(nulls)), n=len(nulls))
    log('null', out['null_within_cell_shuffle'])
    (RES / 'h66_verify_data.json').write_text(json.dumps(out, indent=1)); log('wrote h66_verify_data.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', required=True, choices=['edits', 'summ', 'w', 'data'])
    ap.add_argument('--per', type=int, default=4000); ap.add_argument('--boot', type=int, default=1000)
    ap.add_argument('--boot_lr', type=int, default=40); ap.add_argument('--ntrain', type=int, default=60000)
    a = ap.parse_args()
    dict(edits=step_edits, summ=lambda a: summarize_edits(), w=step_w, data=step_data)[a.step](a)
