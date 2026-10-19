"""H2 (loop D): classical references on the one-signal datasets (Top Tagging Landscape, Quark-Gluon), on the full test sets.

Run (conda python: numba, fastjet for the cross-check):
  AI_PY=conda ./job.sh d-h2-class 4 12 'python -m interp.h2_onesignal_setup --step all'
Steps: top (observables on all 404k test jets + 100k random val jets [seed 0] for the likelihood-ratio fit), qg (observables on
all 200k test jets), lnet (released LorentzNet per-jet test scores, score.npy, aligned to our test rows), stats (ROC metrics with
bootstrap), fig. Writes $AI_RESULTS/physics/h2_top_obs_<split>.npz, h2_qg_obs.npz, h2_classical.json, h2_fig_*.png.

Observables (all constituents of the stored jet; rapidity y and azimuth phi from the four-vectors):
  top  m     invariant mass of the constituent sum [GeV]
       mSD   soft-drop mass: Cambridge/Aachen reclustering (pairwise only, i.e. R -> infinity), declustering from the top,
             keep the step where min(pT1, pT2)/(pT1 + pT2) > zcut = 0.1 (beta = 0), else follow the harder branch
       tauN  N-subjettiness, beta = 1, R0 = 0.8, axes = exclusive kT (pairwise) with winner-take-all pT recombination;
             tau_N = sum_i pT_i min_k dR(i, a_k) / (R0 sum_i pT_i); tau32 = tau3/tau2, tau21 = tau2/tau1
       mW    three exclusive-kT subjets with E-scheme recombination; the pair mass closest to 80.4 GeV; also m12min (smallest
             pair mass) and m123 (three-subjet mass)
  qg   N     constituent multiplicity (all particles; Nch = charged only by PDG id)
       pTD   sqrt(sum pT_i^2) / sum pT_i
       girth sum pT_i dR_i / sum pT_i, dR_i to the stored jet axis (stored deta, dphi)
       C1    C1^(beta=0.2) = sum_{i<j} z_i z_j dR_ij^0.2, z_i = pT_i / sum pT (stored deta, dphi)
Working points (top): CMS-like 105 < mSD < 210 GeV and tau32 < 0.54 (tight) or < 0.65 (loose); the same with 65 < mW < 95 GeV.
Scores for single-observable ROCs: -|x - x0| for window observables (x0 = 172.5 GeV for m, mSD; 80.4 GeV for mW), -tau32,
-tau21; window+tau32 = -tau32 inside 105 < mSD < 210 (else -inf); LR2 = binned signal/background likelihood ratio of
(mSD, tau32), LR3 = of (mSD, tau32, mW), histograms filled on the 100k val jets, evaluated on test. QG: signal = quark; score
-N, -Nch, +pTD, -girth, -C1.
Metrics: AUC, background rejection 1/eps_B at eps_S = 0.3 and 0.5, each with a 95 % bootstrap interval over test jets (B = 200).
"""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
MW, MT = 80.4, 172.5
B = 200

try:
    from numba import njit, prange
except ImportError:  # local import check only
    njit = lambda *a, **k: (lambda f: f); prange = range


@njit(cache=False)
def _cluster(px, py, pz, E, n, p, wta, nstop, tree):
    """pairwise sequential recombination of n particles, distance min(pT_i^2p, pT_j^2p) dR_ij^2, until nstop objects remain.
    wta: winner-take-all pT scheme (pT added, direction of the harder), else E-scheme. Returns the (px, py, pz, E, pt, y, phi)
    of the remaining nstop objects at each count 3, 2, 1 (rows 0-2, 3-4, 5) and, if tree, the merge tree (node 4-vectors,
    children) for soft drop."""
    M = 2 * n
    vx = np.zeros(M); vy = np.zeros(M); vz = np.zeros(M); vE = np.zeros(M); pt = np.zeros(M); yy = np.zeros(M); ph = np.zeros(M)
    c1 = -np.ones(M, np.int64); c2 = -np.ones(M, np.int64); alive = np.zeros(M, np.bool_)
    for i in range(n):
        vx[i] = px[i]; vy[i] = py[i]; vz[i] = pz[i]; vE[i] = E[i]; pt[i] = np.hypot(px[i], py[i])
        ph[i] = np.arctan2(py[i], px[i]); e = max(E[i], abs(pz[i]) + 1e-9); yy[i] = 0.5 * np.log((e + pz[i]) / (e - pz[i]))
        alive[i] = True
    res = np.zeros((6, 7)); nxt = n; cnt = n
    while True:
        if cnt <= 3:
            base = 0 if cnt == 3 else (3 if cnt == 2 else 5); k = 0
            for a in range(nxt):
                if alive[a]:
                    res[base + k, 0] = vx[a]; res[base + k, 1] = vy[a]; res[base + k, 2] = vz[a]; res[base + k, 3] = vE[a]
                    res[base + k, 4] = pt[a]; res[base + k, 5] = yy[a]; res[base + k, 6] = ph[a]; k += 1
        if cnt <= nstop: break
        best = 1e300; bi = -1; bj = -1
        for a in range(nxt):
            if not alive[a]: continue
            for b in range(a + 1, nxt):
                if not alive[b]: continue
                dphi = abs(ph[a] - ph[b])
                if dphi > np.pi: dphi = 2 * np.pi - dphi
                dr2 = (yy[a] - yy[b]) ** 2 + dphi ** 2
                if p == 0: d = dr2
                else: d = min(pt[a] ** (2 * p), pt[b] ** (2 * p)) * dr2
                if d < best: best = d; bi = a; bj = b
        k = nxt; nxt += 1
        alive[bi] = False; alive[bj] = False; alive[k] = True; c1[k] = bi; c2[k] = bj
        vx[k] = vx[bi] + vx[bj]; vy[k] = vy[bi] + vy[bj]; vz[k] = vz[bi] + vz[bj]; vE[k] = vE[bi] + vE[bj]
        if wta:
            h = bi if pt[bi] >= pt[bj] else bj
            pt[k] = pt[bi] + pt[bj]; yy[k] = yy[h]; ph[k] = ph[h]
        else:
            pt[k] = np.hypot(vx[k], vy[k]); ph[k] = np.arctan2(vy[k], vx[k])
            e = max(vE[k], abs(vz[k]) + 1e-9); yy[k] = 0.5 * np.log((e + vz[k]) / (e - vz[k]))
        cnt -= 1
    return res, vx, vy, vz, vE, pt, c1, c2, nxt


@njit(cache=False)
def _m(px, py, pz, E):
    return np.sqrt(max(E * E - px * px - py * py - pz * pz, 0.0))


@njit(parallel=True, cache=False)
def top_obs(F, nn, zcut, R0):
    """F: (J, L, 6) px, py, pz, E, deta, dphi; nn: multiplicities. Returns (J, 10): m, mSD, tau1, tau2, tau3, mW, m12min,
    m123, n_used, sd_ok"""
    J = F.shape[0]; out = np.zeros((J, 10))
    for j in prange(J):
        n = nn[j]
        px = F[j, :n, 0].astype(np.float64); py = F[j, :n, 1].astype(np.float64); pz = F[j, :n, 2].astype(np.float64)
        E = F[j, :n, 3].astype(np.float64)
        out[j, 0] = _m(px.sum(), py.sum(), pz.sum(), E.sum()); out[j, 8] = n
        if n < 3: continue
        # soft drop on the C/A tree
        _, vx, vy, vz, vE, pt, c1, c2, nxt = _cluster(px, py, pz, E, n, 0, False, 1, True)
        node = nxt - 1
        while True:
            a = c1[node]; b = c2[node]
            if a < 0: break
            if min(pt[a], pt[b]) / (pt[a] + pt[b]) > zcut:
                out[j, 9] = 1; break
            node = a if pt[a] >= pt[b] else b
        out[j, 1] = _m(vx[node], vy[node], vz[node], vE[node])
        # WTA kT axes -> tau_N
        rw, _, _, _, _, _, _, _, _ = _cluster(px, py, pz, E, n, 1, True, 1, False)
        ptp = np.hypot(px, py); yp = np.zeros(n); php = np.arctan2(py, px)
        for i in range(n):
            e = max(E[i], abs(pz[i]) + 1e-9); yp[i] = 0.5 * np.log((e + pz[i]) / (e - pz[i]))
        S = ptp.sum() * R0
        for N, base in ((3, 0), (2, 3), (1, 5)):
            t = 0.0
            for i in range(n):
                dm = 1e9
                for k in range(N):
                    dphi = abs(php[i] - rw[base + k, 6])
                    if dphi > np.pi: dphi = 2 * np.pi - dphi
                    d = np.sqrt((yp[i] - rw[base + k, 5]) ** 2 + dphi ** 2)
                    if d < dm: dm = d
                t += ptp[i] * dm
            out[j, 2 + (N - 1)] = t / S
        # E-scheme exclusive kT 3 subjets -> pair masses
        re, _, _, _, _, _, _, _, _ = _cluster(px, py, pz, E, n, 1, False, 3, False)
        best = 1e9; mmin = 1e9
        for a in range(3):
            for b in range(a + 1, 3):
                mp = _m(re[a, 0] + re[b, 0], re[a, 1] + re[b, 1], re[a, 2] + re[b, 2], re[a, 3] + re[b, 3])
                if abs(mp - 80.4) < abs(best - 80.4): best = mp
                if mp < mmin: mmin = mp
        out[j, 5] = best; out[j, 6] = mmin
        out[j, 7] = _m(re[0:3, 0].sum(), re[0:3, 1].sum(), re[0:3, 2].sum(), re[0:3, 3].sum())
    return out


@njit(parallel=True, cache=False)
def c1_beta(F, nn, beta):
    J = F.shape[0]; out = np.zeros(J)
    for j in prange(J):
        n = nn[j]; pt = np.hypot(F[j, :n, 0].astype(np.float64), F[j, :n, 1].astype(np.float64)); S = pt.sum()
        if S <= 0: continue
        s = 0.0
        for a in range(n):
            for b in range(a + 1, n):
                dphi = abs(F[j, a, 5] - F[j, b, 5])
                if dphi > np.pi: dphi = 2 * np.pi - dphi
                dr = np.sqrt((F[j, a, 4] - F[j, b, 4]) ** 2 + dphi ** 2)
                s += pt[a] * pt[b] * dr ** beta
        out[j] = s / S ** 2
    return out


def _chunks(F, nn, fn, step=20_000):
    res = []
    for s in range(0, len(nn), step):
        res.append(fn(np.ascontiguousarray(F[s:s + step]), np.ascontiguousarray(nn[s:s + step])))
        log(f'    {min(s + step, len(nn))}/{len(nn)}')
    return np.concatenate(res)


TOPK = ['m', 'mSD', 'tau1', 'tau2', 'tau3', 'mW', 'm12min', 'm123', 'n', 'sd_ok']


def step_top():
    from interp import data_top as dt
    for split, rows in (('test', None), ('val', 100_000)):
        f = OUT / f'h2_top_obs_{split}.npz'
        if f.exists(): log(f'{f.name} exists'); continue
        D = dt.load_split(split); y = np.asarray(D['label']); jet = np.asarray(D['jet']); nn = jet[:, 5].astype(np.int64)
        idx = np.arange(len(y)) if rows is None else np.sort(np.random.default_rng(0).choice(len(y), rows, replace=False))
        log(f'top {split}: {len(idx)} jets, L = {D["full"].shape[1]}, signal fraction {y[idx].mean():.4f}')
        nnL = np.minimum(nn[idx], D['full'].shape[1])
        Fsel = D['full'] if rows is None else np.asarray(D['full'][idx])
        o = _chunks(Fsel, nnL, lambda a, b: top_obs(a, b, 0.1, 0.8))
        np.savez(f, idx=idx, label=y[idx], jet=jet[idx], **{k: o[:, i] for i, k in enumerate(TOPK)})
        log(f'  wrote {f}; m vs stored jet_mass max |diff| {np.nanmax(np.abs(o[:, 0] - jet[idx, 4])):.3f} GeV ({int(np.isnan(jet[idx, 4]).sum())} stored NaN)')


def step_crosscheck(nj=300):
    """fastjet (3.4.1.2, SWIG interface) on nj random test jets: tau32 with exclusive-kT WTA axes and mSD by declustering"""
    import fastjet as fj
    from interp import data_top as dt
    S = dt.sample(nj // 2, 7); F = S['full']; nn = S['jet'][:, 5].astype(int)
    mine = top_obs(np.ascontiguousarray(F), np.ascontiguousarray(nn.astype(np.int64)), 0.1, 0.8)
    ref = np.zeros((len(nn), 3))
    for j in range(len(nn)):
        ps = [fj.PseudoJet(*map(float, F[j, i, :4])) for i in range(nn[j])]
        cs = fj.ClusterSequence(ps, fj.JetDefinition(fj.kt_algorithm, 999.0, fj.WTA_pt_scheme))
        taus = []
        for N in (1, 2, 3):
            ax = cs.exclusive_jets(N)
            t = sum(p.pt() * min(p.delta_R(a) for a in ax) for p in ps) / (0.8 * sum(p.pt() for p in ps)); taus.append(t)
        ref[j, 0] = taus[2] / taus[1]
        ca = fj.ClusterSequence(ps, fj.JetDefinition(fj.cambridge_algorithm, 999.0))
        jj = ca.inclusive_jets()[0] if len(ca.inclusive_jets()) == 1 else max(ca.inclusive_jets(), key=lambda x: x.pt())
        while True:
            p1, p2 = fj.PseudoJet(), fj.PseudoJet()
            if not jj.has_parents(p1, p2): break
            if min(p1.pt(), p2.pt()) / (p1.pt() + p2.pt()) > 0.1: break
            jj = p1 if p1.pt() >= p2.pt() else p2
        ref[j, 1] = jj.m()
        e3 = fj.ClusterSequence(ps, fj.JetDefinition(fj.kt_algorithm, 999.0)).exclusive_jets(3)
        pm = [(e3[a] + e3[b]).m() for a in range(3) for b in range(a + 1, 3)]
        ref[j, 2] = min(pm, key=lambda v: abs(v - MW))
    mt32 = mine[:, 4] / mine[:, 3]
    r = dict(n=len(nn), tau32_max_abs_diff=float(np.max(np.abs(mt32 - ref[:, 0]))),
             tau32_frac_agree_1em4=float(np.mean(np.abs(mt32 - ref[:, 0]) < 1e-4)),
             mSD_max_abs_diff=float(np.max(np.abs(mine[:, 1] - ref[:, 1]))), mSD_frac_agree_0p1GeV=float(np.mean(np.abs(mine[:, 1] - ref[:, 1]) < 0.1)),
             mW_frac_agree_0p1GeV=float(np.mean(np.abs(mine[:, 5] - ref[:, 2]) < 0.1)))
    log('fastjet cross-check', r); return r


def step_qg():
    from interp import data_qg as dq
    f = OUT / 'h2_qg_obs.npz'
    if f.exists(): log(f'{f.name} exists'); return
    D = dq.load_split('test'); y = np.asarray(D['label']); jet = np.asarray(D['jet']); F = np.asarray(D['full']); pid = np.asarray(D['pid'])
    real = F[..., 3] > 0; pt = np.hypot(F[..., 0], F[..., 1]) * real
    N = real.sum(1); charged = real & np.isin(np.abs(pid), [11, 13, 211, 321, 2212])
    S = pt.sum(1); pTD = np.sqrt((pt ** 2).sum(1)) / S; dR = np.hypot(F[..., 4], F[..., 5]); girth = (pt * dR).sum(1) / S
    log(f'qg test: {len(y)} jets, quark fraction {y.mean():.4f}, L = {F.shape[1]}; C1 ...')
    C1 = _chunks(F, N.astype(np.int64), lambda a, b: c1_beta(a, b, 0.2), step=50_000)
    np.savez(f, label=y, jet=jet, N=N, Nch=charged.sum(1), pTD=pTD, girth=girth, C1=C1)
    log(f'  wrote {f}')


def _auc(s, y):
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _rej(s, y, eff):
    sig = np.sort(s[y == 1]); thr = sig[int(np.floor((1 - eff) * len(sig)))]
    if thr <= -1e29: return np.nan                       # the cut cannot reach this signal efficiency
    eb = np.mean(s[y == 0] >= thr); return 1 / eb if eb > 0 else np.inf


def _metrics(s, y, rng, effs=(0.3, 0.5), boot=B, auc=True):
    s = np.where(np.isfinite(s), s, -1e30)
    pt = dict(AUC=_auc(s, y) if auc else None, **{f'R{int(e * 100)}': _rej(s, y, e) for e in effs})
    bs = []
    for _ in range(boot):
        i = rng.integers(0, len(y), len(y)); yi, si = y[i], s[i]
        bs.append([_auc(si, yi) if auc else np.nan] + [_rej(si, yi, e) for e in effs])
    bs = np.array(bs)
    ci = {k: [float(np.nanpercentile(bs[:, c], 2.5)), float(np.nanpercentile(bs[:, c], 97.5))] for c, k in enumerate(['AUC'] + [f'R{int(e * 100)}' for e in effs])}
    return dict(point={k: (None if v is None else float(v)) for k, v in pt.items()}, ci95=ci), bs


def _wp(sel, y, rng):
    es, eb = sel[y == 1].mean(), sel[y == 0].mean()
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y)); yi, si = y[i], sel[i]; bs.append([si[yi == 1].mean(), 1 / max(si[yi == 0].mean(), 1e-12)])
    bs = np.array(bs)
    return dict(eps_S=float(es), eps_B=float(eb), rejection=float(1 / eb), eps_S_ci95=np.percentile(bs[:, 0], [2.5, 97.5]).tolist(),
                rejection_ci95=np.percentile(bs[:, 1], [2.5, 97.5]).tolist())


def _lr_fit(X, y, edges):
    hs, _ = np.histogramdd(X[y == 1], bins=edges); hb, _ = np.histogramdd(X[y == 0], bins=edges)
    return np.log((hs + 0.5) / (hs.sum() + 0.5 * hs.size)) - np.log((hb + 0.5) / (hb.sum() + 0.5 * hb.size))


def _lr_eval(H, X, edges):
    ix = tuple(np.clip(np.digitize(X[:, k], edges[k]) - 1, 0, len(edges[k]) - 2) for k in range(X.shape[1]))
    return H[ix]


def step_stats():
    rng = np.random.default_rng(1); R = {}
    T = dict(np.load(OUT / 'h2_top_obs_test.npz')); V = dict(np.load(OUT / 'h2_top_obs_val.npz')); y = T['label'].astype(int)
    for d in (T, V):
        d['tau32'] = d['tau3'] / np.maximum(d['tau2'], 1e-12); d['tau21'] = d['tau2'] / np.maximum(d['tau1'], 1e-12)
    win = (T['mSD'] > 105) & (T['mSD'] < 210); wW = (T['mW'] > 65) & (T['mW'] < 95)
    scores = {'m': -np.abs(T['m'] - MT), 'mSD': -np.abs(T['mSD'] - MT), 'tau32': -T['tau32'], 'tau21': -T['tau21'],
              'mW': -np.abs(T['mW'] - MW), 'mSD window + tau32': np.where(win, -T['tau32'], -np.inf),
              'mSD window + mW window + tau32': np.where(win & wW, -T['tau32'], -np.inf)}
    e2 = [np.linspace(0, 300, 31), np.linspace(0, 1, 26)]; e3 = e2[:1] + e2[1:] + [np.linspace(0, 200, 21)]
    Xv2 = np.stack([V['mSD'], V['tau32']], 1); Xv3 = np.stack([V['mSD'], V['tau32'], V['mW']], 1)
    H2 = _lr_fit(Xv2, V['label'], e2); H3 = _lr_fit(Xv3, V['label'], e3)
    scores['LR(mSD, tau32)'] = _lr_eval(H2, np.stack([T['mSD'], T['tau32']], 1), e2)
    scores['LR(mSD, tau32, mW)'] = _lr_eval(H3, np.stack([T['mSD'], T['tau32'], T['mW']], 1), e3)
    rt, boots = {}, {}
    for k, s in scores.items():
        rt[k], boots[k] = _metrics(s, y, np.random.default_rng(11)); log('top', k, rt[k]['point'])
    wps = {}
    for cut in (0.54, 0.65):
        wps[f'105<mSD<210 & tau32<{cut}'] = _wp(win & (T['tau32'] < cut), y, rng)
        wps[f'105<mSD<210 & 65<mW<95 & tau32<{cut}'] = _wp(win & wW & (T['tau32'] < cut), y, rng)
    wps['105<mSD<210'] = _wp(win, y, rng); wps['65<mW<95'] = _wp(wW, y, rng)
    # paired comparison LR3 vs LR2 (same bootstrap resamples: same rng seed 11 in _metrics)
    cmp = {}
    for c, e in ((1, 'R30'), (2, 'R50')):
        d = np.log(boots['LR(mSD, tau32, mW)'][:, c]) - np.log(boots['LR(mSD, tau32)'][:, c]); d = d[np.isfinite(d)]
        p = 2 * min(np.mean(d <= 0), np.mean(d >= 0)); cmp[e] = dict(log_ratio=float(np.median(d)), ci95=np.percentile(d, [2.5, 97.5]).tolist(),
                                                                  p_two_sided=max(float(p), 1 / len(d)))
    ps = [cmp[e]['p_two_sided'] for e in cmp]; o = np.argsort(ps); adj = np.minimum(1, np.maximum.accumulate([ps[i] * (len(ps) - r) for r, i in enumerate(o)]))
    for r, i in enumerate(o): cmp[list(cmp)[i]]['p_holm'] = float(adj[r])
    sig, bkg = y == 1, y == 0
    dist = {k: dict(signal_median=float(np.median(T[k][sig])), background_median=float(np.median(T[k][bkg])),
                    signal_iqr=np.percentile(T[k][sig], [25, 75]).tolist(), background_iqr=np.percentile(T[k][bkg], [25, 75]).tolist())
            for k in ('m', 'mSD', 'tau32', 'tau21', 'mW', 'm12min', 'm123')}
    R['top'] = dict(N_test=int(len(y)), N_signal=int(sig.sum()), N_val_for_LR=int(len(V['label'])), roc=rt, working_points=wps,
                    LR3_vs_LR2=cmp, distributions=dist, sd_groomed_fraction=float(T['sd_ok'].mean()),
                    m_vs_stored_jet_mass_max_abs=float(np.nanmax(np.abs(T['m'] - T['jet'][:, 4]))),
                    stored_jet_mass_nan=int(np.isnan(T['jet'][:, 4]).sum()))
    # QG
    Q = dict(np.load(OUT / 'h2_qg_obs.npz')); yq = Q['label'].astype(int)
    qs = {'N': -Q['N'].astype(float), 'Nch': -Q['Nch'].astype(float), 'pTD': Q['pTD'], 'girth': -Q['girth'], 'C1(beta=0.2)': -Q['C1']}
    rq = {}
    for k, s in qs.items():
        rq[k], _ = _metrics(s, yq, np.random.default_rng(12)); log('qg', k, rq[k]['point'])
    cas = {}
    for k in ('N', 'Nch'):
        g, q = Q[k][yq == 0].astype(float), Q[k][yq == 1].astype(float); r0 = g.mean() / q.mean()
        bs = np.array([rng.choice(g, len(g)).mean() / rng.choice(q, len(q)).mean() for _ in range(1000)])
        # H0: ratio = C_A/C_F = 9/4 ; bootstrap p = fraction of resampled ratios at or beyond 9/4
        cas[k] = dict(mean_gluon=float(g.mean()), mean_quark=float(q.mean()), ratio_g_over_q=float(r0),
                      ratio_ci95=np.percentile(bs, [2.5, 97.5]).tolist(), LO_prediction=2.25,
                      p_H0_ratio_eq_9_4=max(float(np.mean(bs >= 2.25)) * 2, 1 / len(bs)))
    R['qg'] = dict(N_test=int(len(yq)), N_quark=int(yq.sum()), roc=rq, casimir=cas)
    lf = OUT / 'h2_lnet.json'
    if lf.exists(): R['lorentznet_released'] = json.loads(lf.read_text())
    cf = OUT / 'h2_crosscheck.json'
    if cf.exists(): R['fastjet_crosscheck'] = json.loads(cf.read_text())
    R['published'] = dict(top={'ParT': dict(AUC=0.9858, R50=413, R30=1602), 'ParticleNet': dict(AUC=0.9858, R50=397, R30=1615),
                               'PFN': dict(AUC=0.9819, R50=247, R30=888), 'LorentzNet': dict(AUC=0.9868, R50=498, R30=2195)},
                          qg={'ParT_exp(kinpid)': dict(AUC=0.9121, R50=41.3, R30=101.2), 'ParticleNet_exp(kinpid)': dict(AUC=0.9116, R50=39.8, R30=98.6),
                              'PFN_exp': dict(AUC=0.9005, R50=34.7), 'LorentzNet_full': dict(AUC=0.9156, R50=42.4, R30=110.2)},
                          source='Qu, Li, Qian, arXiv:2202.03772, Tables 5-6')
    (OUT / 'h2_classical.json').write_text(json.dumps(R, indent=1))
    log('wrote h2_classical.json')


def step_lnet():
    """released LorentzNet top scores (ext/LorentzNet-release/logs/top/pretrained/score.npy: label, p0, p1 per test jet,
    404,000 rows) vs our test labels (403,999 rows): alignment by label agreement at offsets 0 and 1, then AUC / R30 / R50."""
    s = np.load('/j-jepa-vol/aaron-interp/models/top/lorentznet/score.npy'); T = np.load(OUT / 'h2_top_obs_test.npz'); y = T['label'].astype(int)
    rel = s[:, 0].astype(int); r = dict(n_released=len(rel), n_ours=len(y))
    for off in (0, 1):
        n = min(len(y), len(rel) - off); r[f'label_agree_offset{off}'] = float(np.mean(rel[off:off + n] == y[:n]))
    m, _ = _metrics(s[:, 2], rel, np.random.default_rng(13)); r['metrics_on_released_labels'] = m
    (OUT / 'h2_lnet.json').write_text(json.dumps(r, indent=1)); log('lnet', r)


def step_fig():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    T = dict(np.load(OUT / 'h2_top_obs_test.npz')); Q = dict(np.load(OUT / 'h2_qg_obs.npz')); R = json.loads((OUT / 'h2_classical.json').read_text())
    y = T['label'].astype(int); tau32 = T['tau3'] / np.maximum(T['tau2'], 1e-12)
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    for a, (x, rg, lab) in zip(ax[:3], ((T['mSD'], (0, 300), 'soft-drop mass $m_{SD}$ [GeV]'), (tau32, (0, 1), r'$\tau_{32}$ (WTA $k_T$ axes)'),
                                        (T['mW'], (0, 200), r'best subjet-pair mass $m_W$ [GeV]'))):
        for c, nm, col in ((1, 'top', 'C3'), (0, 'QCD', 'C0')):
            a.hist(x[y == c], bins=60, range=rg, density=True, histtype='step', color=col, label=nm, lw=1.4)
        a.set_xlabel(lab); a.set_ylabel('density'); a.legend(frameon=False)
    ax[1].set_title('Top Landscape test set (404k jets)', fontsize=10)
    yq = Q['label'].astype(int)
    for c, nm, col in ((1, 'quark', 'C2'), (0, 'gluon', 'C1')):
        ax[3].hist(Q['N'][yq == c], bins=np.arange(0, 140, 2), density=True, histtype='step', color=col, lw=1.4, label=nm)
    cs = R['qg']['casimir']['N']
    ax[3].set_xlabel('constituent multiplicity $N$'); ax[3].set_ylabel('density'); ax[3].legend(frameon=False)
    ax[3].set_title(f"QG test (200k): $\\langle N_g\\rangle/\\langle N_q\\rangle$ = {cs['ratio_g_over_q']:.2f} (LO 9/4)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / 'h2_fig_observables.png', dpi=120); log('fig observables')
    # rejection bars
    rt = R['top']['roc']; keys = list(rt)
    fig, ax = plt.subplots(figsize=(9, 4))
    v = [rt[k]['point']['R30'] for k in keys]; lo = [rt[k]['ci95']['R30'][0] for k in keys]; hi = [rt[k]['ci95']['R30'][1] for k in keys]
    ax.barh(range(len(keys)), v, xerr=[np.array(v) - lo, np.array(hi) - v], color='C7')
    ax.axvline(1602, color='C3', ls='--', label='ParT (published) 1602'); ax.axvline(2195, color='C1', ls=':', label='LorentzNet (published) 2195')
    ax.set_xscale('log'); ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys); ax.set_xlabel(r'QCD rejection $1/\varepsilon_B$ at $\varepsilon_S = 0.3$ (95 % CI)')
    ax.legend(frameon=False, fontsize=8, loc='lower right'); ax.set_title('QCD rejection at 30 % top efficiency: classical (bars), published networks (lines)', fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / 'h2_fig_top_rejection.png', dpi=120); log('fig rejection')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--step', default='all'); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    steps = ['top', 'crosscheck', 'qg', 'lnet', 'stats', 'fig'] if a.step == 'all' else a.step.split(',')
    for st in steps:
        log(f'== step {st}')
        if st == 'crosscheck':
            try: (OUT / 'h2_crosscheck.json').write_text(json.dumps(step_crosscheck(), indent=1))
            except Exception as e: log(f'crosscheck failed: {type(e).__name__}: {e}')
        else: globals()[f'step_{st}']()
    log('done')
