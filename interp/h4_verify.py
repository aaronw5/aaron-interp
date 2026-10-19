"""H4 verification (loop D): rerun the W-pair edit on a fresh sample and add three controls.

Fresh sample: 5,000 top + 5,000 QCD test jets (seed 611) and 90,000 QCD test jets (seed 612; all remaining QCD test rows number about 92k), drawn at random from the
Top Landscape test rows NOT used by H4 (seeds 601 / 602). Functions (subjets, edit, controls, observables, network
forward passes) are imported unchanged from interp.h4_top_wmass.
Edits / targets:
  T60, T100  replication of H4 (top jets, 65 < mW < 95, acceptance as H4: |dm| < 0.5, |dtau32| < 0.02, |mW - t| < 5).
  T80        new: top jets with 6 < |mW - 80.4| < 15 moved ONTO 80.4 GeV (|mW - 80.4| < 2 after). An mW-window reading
             predicts a score rise (LR3 / MLP3 direction); a 'any deformation of a top jet lowers the score' reading
             predicts a fall.
  Q80        replication, split into Q80up (mW0 50-70, pair opened) and Q80dn (mW0 90-115, pair closed): an mW-window
             reading predicts a rise in both; a monotone dependence on the pair opening predicts opposite signs.
Controls added:
  MLPx   MLP(64, 64) on (mSD, tau1, tau2, tau3, tau32, tau21, multiplicity, sorted subjet pT fractions, sorted subjet masses),
         no pair mass, no pair angle; fitted on 100,000 random val jets (seed 613). It sees the side effects of the edit
         (subjet sizes / masses, tau21) but not mW. MLPxW = the same plus mW (positive control on the same features).
  h      the subjet-size factor h of the W edit vs the N control, and per-jet regressions of each network's
         Delta score on (Delta mW toward/away from 80.4, Delta sum of subjet masses, Delta tau21).
Outputs: $AI_RESULTS/physics/h4_verify.json, h4_verify/ (npz). Run: ./part_job.sh d-h4-verify 'python -m interp.h4_verify'
"""
import json
import os
import time
from pathlib import Path
import numpy as np
from interp import h4_top_wmass as H
from interp import data_top as dt

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
OUT2 = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
OUT = OUT2 / 'h4_verify'
B = 2000
NCPU = int(os.environ.get('OMP_NUM_THREADS', '4'))


def fresh(split, per_class, seed, classes, exclude):
    y = np.load(dt.DST / split / 'label.npy'); rng = np.random.default_rng(seed)
    ok = np.ones(len(y), bool); ok[exclude] = False
    return np.sort(np.concatenate([rng.choice(np.flatnonzero((y == c) & ok), per_class, replace=False) for c in classes])), y


def edit_set(P4id, nn, rows, t):
    import multiprocessing as mp
    F = np.zeros((len(rows), P4id.shape[1], 4), np.float64); F[:] = P4id[rows]; LAB = H.kt3_batch(F, nn[rows])
    jobs = [(int(j), P4id[j, :nn[j]], LAB[ii, :nn[j]], float(t)) for ii, j in enumerate(rows)]
    with mp.get_context('fork').Pool(NCPU) as pool:
        res = pool.map(H._work, jobs, chunksize=8)
    out = {}
    for k in 'WNR':
        rr = [(j, r[k]) for j, r in zip(rows, res) if r.get(k) is not None]
        P = np.zeros((len(rr), P4id.shape[1], 4), np.float32)
        for i, (_, d) in enumerate(rr): P[i, :len(d['P'])] = d['P']
        out[k] = dict(rows=np.array([j for j, _ in rr], np.int64), P=P, h=np.array([d['meta'].get('h', 1.0) for _, d in rr]),
                      mW0=np.array([d['meta']['mW0'] for _, d in rr]))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.array([[1.0, 0, 5, 6], [0, 1.0, 2, 3], [0.5, 0.5, 1, 2]]); H._tau32(z[:, 0].copy(), z[:, 1].copy(), z[:, 2].copy(), z[:, 3].copy(), 3)
    # ------------------------------------------------------------------ fresh rows, excluding the H4 rows
    yt = np.load(dt.DST / 'test' / 'label.npy'); rng0 = np.random.default_rng(601)
    oldT = np.concatenate([rng0.choice(np.flatnonzero(yt == c), 5000, replace=False) for c in (0, 1)])
    oldQ = np.random.default_rng(602).choice(np.flatnonzero(yt == 0), 100000, replace=False)
    excl = np.concatenate([oldT, oldQ])
    idxT, _ = fresh('test', 5000, 611, (0, 1), excl); idxQ, _ = fresh('test', 90000, 612, (0,), np.concatenate([excl, idxT]))
    log(f'fresh rows: T {len(idxT)}, Q {len(idxQ)}; overlap with H4 rows: {len(np.intersect1d(idxT, excl))}, {len(np.intersect1d(idxQ, excl))}')
    full = H._full_test(); jet = np.load(dt.DST / 'test' / 'jet.npy')
    sets = {}
    for name, idx in (('T', idxT), ('Q', idxQ)):
        nn = np.minimum(jet[idx, 5].astype(np.int64), full.shape[1]); P4id = H._identity(full[idx][..., :4].astype(np.float32), nn).astype(np.float32)
        O = H._obs(P4id, nn); sets[name] = dict(idx=idx, y=yt[idx].astype(int), nn=nn, P4=P4id, O=O)
        log(f'set {name}: {len(nn)} jets, observables done')
    # ------------------------------------------------------------------ edits
    E = {}
    S = sets['T']; O = S['O']; win = (O['mSD'] > 105) & (O['mSD'] < 210) & (S['nn'] >= 3) & (S['y'] == 1)
    for t in (60.0, 100.0):
        E[f'T{int(t)}'] = dict(set='T', t=t, tolW=5.0, edit=edit_set(S['P4'], S['nn'], np.flatnonzero(win & (O['mW'] > 65) & (O['mW'] < 95)), t)); log(f'T{int(t)} edited')
    d80 = np.abs(O['mW'] - H.MWREF); E['T80'] = dict(set='T', t=H.MWREF, tolW=2.0, edit=edit_set(S['P4'], S['nn'], np.flatnonzero(win & (d80 > 6) & (d80 < 15)), H.MWREF)); log('T80 edited')
    S = sets['Q']; O = S['O']; winq = (O['mSD'] > 105) & (O['mSD'] < 210) & (S['nn'] >= 3)
    E['Q80'] = dict(set='Q', t=H.MWREF, tolW=5.0, edit=edit_set(S['P4'], S['nn'], np.flatnonzero(winq & (((O['mW'] > 50) & (O['mW'] < 70)) | ((O['mW'] > 90) & (O['mW'] < 115)))), H.MWREF)); log('Q80 edited')
    # acceptance (H4 rules) and analysis sets
    for key, e in E.items():
        S = sets[e['set']]; O0 = S['O']; acc = {}
        for k in 'WN':
            ed = e['edit'][k]; Oe = H._obs(ed['P'], S['nn'][ed['rows']]); ed['O'] = Oe; rr = ed['rows']
            dm = np.abs(Oe['m'] - O0['m'][rr]); dtau = np.abs(Oe['tau32'] - O0['tau32'][rr])
            okW = np.abs(Oe['mW'] - e['t']) < e['tolW'] if k == 'W' else np.abs(Oe['mW'] - O0['mW'][rr]) < 2
            acc[k] = set(rr[(dm < 0.5) & (dtau < 0.02) & okW].tolist())
        rows = np.array(sorted(acc['W'] & acc['N']), np.int64); e['rows'] = rows
        e['pos'] = {k: np.searchsorted(e['edit'][k]['rows'], rows) for k in 'WNR'}
        e['n'] = dict(W_conv=int(len(e['edit']['W']['rows'])), N_conv=int(len(e['edit']['N']['rows'])), W_acc=len(acc['W']), N_acc=len(acc['N']), analysis=int(len(rows)))
        log(key, e['n'])
    # ------------------------------------------------------------------ substructure features (analysis rows: id, W, N)
    for key, e in E.items():
        S = sets[e['set']]; rows = e['rows']; nn = S['nn'][rows]
        e['X'] = {'id': H._features(S['P4'][rows], nn)}
        for k in 'WNR': e['X'][k] = H._features(e['edit'][k]['P'][e['pos'][k]], nn)
    # ------------------------------------------------------------------ classical models
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    Vv = dict(np.load(OUT2 / 'h2_top_obs_val.npz')); Vv['tau32'] = Vv['tau3'] / np.maximum(Vv['tau2'], 1e-12); yv = Vv['label'].astype(int)
    e2 = [np.linspace(0, 300, 31), np.linspace(0, 1, 26)]; e3 = e2 + [np.linspace(0, 200, 21)]
    H2 = H._lr_fit(np.stack([Vv['mSD'], Vv['tau32']], 1), yv, e2); H3 = H._lr_fit(np.stack([Vv['mSD'], Vv['tau32'], Vv['mW']], 1), yv, e3)
    X3 = np.stack([Vv['mSD'], Vv['tau32'], Vv['mW']], 1); sc3 = StandardScaler().fit(X3)
    mlp3 = MLPClassifier(hidden_layer_sizes=(64, 64), early_stopping=True, random_state=0, max_iter=200).fit(sc3.transform(X3), yv)
    yval = np.load(dt.DST / 'val' / 'label.npy'); rv = np.random.default_rng(613)
    iv = np.sort(np.concatenate([rv.choice(np.flatnonzero(yval == c), 50000, replace=False) for c in (0, 1)]))
    t = time.time(); fv = np.load(dt.DST / 'val' / 'full.npy'); Pv = fv[iv][..., :4].astype(np.float32); del fv
    nv = np.minimum(np.load(dt.DST / 'val' / 'jet.npy')[iv, 5].astype(np.int64), Pv.shape[1]); log(f'val rows read in {time.time() - t:.0f}s')
    Xv, Ov = H._features(Pv, nv); del Pv

    def xsel(X, O, withW):
        c = [X[:, 0], X[:, 1], X[:, 2], X[:, 3], O['tau32'], X[:, 4], X[:, 7], X[:, 11], X[:, 12], X[:, 13], X[:, 14], X[:, 15], X[:, 16]]
        return np.column_stack(c + ([O['mW']] if withW else []))
    yvv = yval[iv].astype(int); mods = {}
    for nm, w in (('MLPx', False), ('MLPxW', True)):
        Xf = xsel(Xv, Ov, w); scx = StandardScaler().fit(Xf)
        m = MLPClassifier(hidden_layer_sizes=(64, 64), early_stopping=True, random_state=0, max_iter=200).fit(scx.transform(Xf), yvv)
        mods[nm] = (m, scx, w); log(f'{nm}: {m.n_iter_} iterations, val score {m.best_validation_score_:.4f}')
    lo = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))

    def classical(X, O):
        r = {'LR2': H._lr_eval(H2, np.stack([O['mSD'], O['tau32']], 1), e2), 'LR3': H._lr_eval(H3, np.stack([O['mSD'], O['tau32'], O['mW']], 1), e3),
             'MLP3': lo(mlp3.predict_proba(sc3.transform(np.stack([O['mSD'], O['tau32'], O['mW']], 1)))[:, 1])}
        for nm, (m, scx, w) in mods.items(): r[nm] = lo(m.predict_proba(scx.transform(xsel(X, O, w)))[:, 1])
        return r
    # ------------------------------------------------------------------ scores. T set: the full 10k sample per variant (rows substituted);
    # Q set: only the Q80 eligible rows. Thresholds at eps_S 0.3 / 0.5 from the 5,000 fresh unedited tops.
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    nets = {'ParT': H.WeaverNet('ParT', OUT2 / 'h4_ckpt_part_kin.pt', dev), 'PN': H.WeaverNet('PN', OUT2 / 'h4_ckpt_pnet_kin.pt', dev),
            'LNet': H.LNet(dev, mass64=True)}
    flows = {a.upper(): H.Flow(a) for a in ('pfn', 'efn')}

    def score_all(P4, nn, order):
        r = {}
        for k, n in nets.items(): r[k] = n.score(P4, nn, order) if k == 'LNet' else n.score(P4, nn)
        k3 = np.concatenate([H.kin3_of(P4[s:s + 5000], nn[s:s + 5000]) for s in range(0, len(nn), 5000)])
        for k, f in flows.items(): r[k] = f.score(k3)
        return r
    SC = {}
    S = sets['T']; ordT = np.random.default_rng(605).permutation(len(S['nn']))
    SC['T|id'] = score_all(S['P4'], S['nn'], ordT)
    Xid_full, _ = H._features(S['P4'], S['nn']); SC['T|id'].update(classical(Xid_full, S['O']))
    perf = {}
    for k, s in SC['T|id'].items():
        perf[k] = dict(AUC=float(H._auc(s, S['y'])), acc=float(((s > 0) == (S['y'] == 1)).mean()) if not k.startswith('LR') else None)
    log('unedited', perf)
    thr = {k: dict(eS30=float(np.quantile(s[S['y'] == 1], 0.7)), eS50=float(np.quantile(s[S['y'] == 1], 0.5))) for k, s in SC['T|id'].items()}
    Sq = sets['Q']; qrows = np.unique(np.concatenate([E['Q80']['edit']['W']['rows'], E['Q80']['edit']['N']['rows']]))
    qpos = lambda r: np.searchsorted(qrows, r); ordQ = np.random.default_rng(605).permutation(len(qrows))
    SC['Q|id'] = score_all(Sq['P4'][qrows], Sq['nn'][qrows], ordQ)
    Xq, _ = H._features(Sq['P4'][qrows], Sq['nn'][qrows]); Oq = {k: v[qrows] for k, v in Sq['O'].items()}; SC['Q|id'].update(classical(Xq, Oq))
    for key, e in E.items():
        for k in 'WNR':
            ed = e['edit'][k]
            if e['set'] == 'T':
                P = sets['T']['P4'].copy(); P[ed['rows']] = ed['P']; s = score_all(P, sets['T']['nn'], ordT)
                s = {m: v[e['rows']] for m, v in s.items()}
            else:
                P = Sq['P4'][qrows].copy(); P[qpos(ed['rows'])] = ed['P']; s = score_all(P, Sq['nn'][qrows], ordQ)
                s = {m: v[qpos(e['rows'])] for m, v in s.items()}
            Xe, Oe = e['X'][k]; s.update(classical(Xe, Oe)); SC[f'{key}|{k}'] = s
        base = SC['T|id'] if e['set'] == 'T' else {m: v for m, v in SC['Q|id'].items()}
        e['s0'] = {m: (v[e['rows']] if e['set'] == 'T' else v[qpos(e['rows'])]) for m, v in base.items()}
        log(f'{key} scored')
    # ------------------------------------------------------------------ statistics
    MODELS = ['ParT', 'PN', 'LNet', 'PFN', 'EFN', 'LR2', 'LR3', 'MLP3', 'MLPx', 'MLPxW']
    direction = {'T60': -1, 'T100': -1, 'T80': 1, 'Q80': 1, 'Q80up': 1, 'Q80dn': 1}
    R = dict(sample=dict(T_seed=611, Q_seed=612, n_T=int(len(idxT)), n_Q=int(len(idxQ)), excluded_H4_rows=True), unedited=perf, sets={})
    tests = []
    keys = list(E) + ['Q80up', 'Q80dn']
    for key in keys:
        e = E[key] if key in E else E['Q80']
        sel = np.ones(len(e['rows']), bool)
        if key in ('Q80up', 'Q80dn'):
            mW0 = sets['Q']['O']['mW'][e['rows']]; sel = mW0 < 80.4 if key == 'Q80up' else mW0 > 80.4
        n = int(sel.sum()); r = dict(n_analysis=n, counts=e['n'] if key in E else None, models={})
        if n == 0: R['sets'][key] = r; continue
        X0, O0 = e['X']['id']; XW, OW = e['X']['W']; XN, ON = e['X']['N']
        r['observables'] = {nm: {v: float(np.median((Xv_[:, i] if i is not None else Ov_[q])[sel])) for v, (Xv_, Ov_) in (('id', (X0, O0)), ('W', (XW, OW)), ('N', (XN, ON)))}
                            for nm, i, q in (('mW', None, 'mW'), ('m', None, 'm'), ('tau32', None, 'tau32'), ('tau21', 4, None), ('subm_sum', None, None))
                            if nm != 'subm_sum'}
        r['observables']['subm_sum'] = {v: float(np.median(Xv_[:, 14:17].sum(1)[sel])) for v, Xv_ in (('id', X0), ('W', XW), ('N', XN))}
        r['h'] = {k: dict(median=float(np.median(e['edit'][k]['h'][e['pos'][k]][sel])), p10=float(np.percentile(e['edit'][k]['h'][e['pos'][k]][sel], 10)),
                          p90=float(np.percentile(e['edit'][k]['h'][e['pos'][k]][sel], 90))) for k in 'WN'}
        rng = np.random.default_rng(7); bi = rng.integers(0, n, (B, n))
        for m in MODELS:
            s0 = e['s0'][m][sel]; d = {k: SC[f'{key if key in E else "Q80"}|{k}'][m][sel] - s0 for k in 'WNR'}
            diff = (d['W'] - d['N'])[bi].mean(1); mu = float((d['W'] - d['N']).mean()); sg = direction[key]
            p1 = float((np.sum(sg * diff <= 0) + 1) / (B + 1)); p2 = float(min(1.0, 2 * min((np.sum(diff <= 0) + 1) / (B + 1), (np.sum(diff >= 0) + 1) / (B + 1))))
            M = dict(dW=float(d['W'].mean()), dW_ci95=np.percentile(d['W'][bi].mean(1), [2.5, 97.5]).tolist(), dN=float(d['N'].mean()),
                     dN_ci95=np.percentile(d['N'][bi].mean(1), [2.5, 97.5]).tolist(), dR=float(d['R'].mean()),
                     W_minus_N=mu, ci95=np.percentile(diff, [2.5, 97.5]).tolist(), p_one_sided_mW_direction=p1, p_two_sided=p2)
            if m in thr:
                tq = thr[m]['eS50']; c0 = s0 >= tq; cW = (s0 + d['W']) >= tq; cN = (s0 + d['N']) >= tq
                M['eS50'] = dict(pass_before=float(c0.mean()), pass_W=float(cW.mean()), pass_N=float(cN.mean()), mcnemar_W_vs_N=H._mcnemar(cN, cW))
            # per-jet regression of the W-edit response on the side effects
            if m in ('ParT', 'PN', 'LNet', 'PFN', 'EFN', 'MLP3', 'MLPx'):
                dist0 = np.abs(O0['mW'][sel] - 80.4); dist1 = np.abs(OW['mW'][sel] - 80.4)
                Z = np.column_stack([np.ones(n), dist1 - dist0, (XW[:, 14:17].sum(1) - X0[:, 14:17].sum(1))[sel], (XW[:, 4] - X0[:, 4])[sel]])
                if n > 10 and np.linalg.matrix_rank(Z) == 4:
                    beta = np.linalg.lstsq(Z, d['W'], rcond=None)[0]
                    bb = np.array([np.linalg.lstsq(Z[i], d['W'][i], rcond=None)[0] for i in bi[:300]])
                    M['regression_dW'] = dict(terms=['const', 'd|mW-80.4| [GeV]', 'd sum subjet mass [GeV]', 'd tau21'], beta=beta.tolist(),
                                              ci95=np.percentile(bb, [2.5, 97.5], 0).T.tolist())
            r['models'][m] = M
            if key not in ('Q80',): tests.append((key, m, p1))
        R['sets'][key] = r
    adj = H._holm([t[2] for t in tests])
    for (key, m, _), pa in zip(tests, adj): R['sets'][key]['models'][m]['p_holm'] = float(pa)
    R['n_tests_holm'] = len(tests); R['holm_family'] = 'sets T60, T100, T80, Q80up, Q80dn x 10 models, one-sided in the mW-window direction'
    (OUT2 / 'h4_verify.json').write_text(json.dumps(R, indent=1, default=float)); log('wrote h4_verify.json')
    for key, r in R['sets'].items():
        log(key, 'n', r['n_analysis'], 'h', r.get('h'))
        for m, M in r['models'].items():
            log(f"  {m:6s} dW {M['dW']:+.3f} dN {M['dN']:+.3f} dR {M['dR']:+.3f} W-N {M['W_minus_N']:+.3f} {np.round(M['ci95'], 3)} p1 {M['p_one_sided_mW_direction']:.3g} "
                f"holm {M.get('p_holm', float('nan')):.3g}" + (f" reg {np.round(M['regression_dW']['beta'], 3)}" if 'regression_dW' in M else ''))


if __name__ == '__main__':
    main()
