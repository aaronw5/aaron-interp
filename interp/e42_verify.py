"""E42 verification: fresh-sample rerun of the W-pair edit on JEDI-linear (official 3-feature n64) plus two controls E42 lacked.

Sample: jedi_core.sample(3000, 4302) (fresh seed); jets whose full_test row was also drawn by the E42 samples (seeds 4201,
  4202, 2000 per class) are removed before any edit.
Edits: interp.h4_verify.edit_set, unchanged (the E42 / H4 definition).  Sets:
  T60, T100  replication (tops, 105 < mSD < 210, 65 < mW < 95; acceptance as E42: |dm| < 0.5, |dtau32| < 0.02, |mW - t| < 5;
             N: |dm| < 0.5, |dtau32| < 0.02, |mW - mW0| < 2).
  T80        new control (as H4_verify on Top Landscape): tops in the mSD window with 6 < |mW0 - 80.4| < 15 GeV moved ONTO
             80.4 GeV (W acceptance |mW - 80.4| < 2).  An mW reader predicts W - N > 0 (top score rises as the W pair
             approaches 80 GeV); a 'deformation of the W pair lowers the score whatever mW does' reading predicts W - N < 0.
Classifier references (t vs q/g, 25,000 + 25,000 hls4ml train-archive jets, dev split excluded, rng 4205, as E42):
  REF   MLP(m, tau32, mW)                                   (E42's mW reader)
  MLPx  MLP(m, mSD, tau1, tau2, tau3, tau21, tau32, n)       (no pair mass: sees only the edits' side effects)
  MLPxW MLPx features + mW                                  (positive control on the same features)
Network: every variant rebuilt about its own E-scheme axis (E42 'rc' build).  Score s = L_t - logsumexp(other four).
Metrics: W - N mean, 95 % jet bootstrap (2,000), sign-flip permutation p (10,000), McNemar on top calls; k_DIL for each
  reference (antisymmetric DIL pairs, as E42, and the through-origin slope); per-jet regression of (dW - dN) on the W - N
  differences of radial input sums (sum r, sum r^2, sum pT r, sum_{rank>=1} (r - 0.2)+, log M2) and tau21, tau1, mSD
  (standardised; intercept with bootstrap CI = W - N not linearly accounted for by these side effects);
  z0 / z10 reads and S58 / S60 / S13 single-unit holds on T60 (share of the W60 response removed); 3 untrained redraws.
Outputs: $AI_RESULTS/jedi/e42_verify.json (the intermediate npz lives in the pod's /tmp only).
Run: AI_PY=conda ./job.sh e42-verify 'python -m interp.e42_verify --part edit && /j-jepa-vol/phat-jet-aaron/env/bin/python -m interp.e42_verify --part jedi'
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

os.environ.setdefault('KERAS_BACKEND', 'jax')
T00 = time.time()
SMOKE = int(os.environ.get('E42V_SMOKE', '0'))
PER = SMOKE or 3000
SEED = 4302
OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'jedi'
TAG = 'e42_verify_smoke' if SMOKE else 'e42_verify'
TMP = Path(os.environ.get('E42V_TMP', '/tmp/e42v'))
SETS = ('T60', 'T100', 'T80')
TARGET = {'T60': 60.0, 'T100': 100.0, 'T80': 80.4}
TOLW = {'T60': 5.0, 'T100': 5.0, 'T80': 2.0}
LAMS = (0.85, 0.92, 0.96, 1.04, 1.08, 1.15)
OBSK = ('m', 'mSD', 'tau1', 'tau2', 'tau3', 'tau21', 'tau32', 'mW', 'n')
REFF = {'REF': ('m', 'tau32', 'mW'), 'MLPx': ('m', 'mSD', 'tau1', 'tau2', 'tau3', 'tau21', 'tau32', 'n'),
        'MLPxW': ('m', 'mSD', 'tau1', 'tau2', 'tau3', 'tau21', 'tau32', 'n', 'mW')}


def log(*a):
    print(f'[{time.time() - T00:7.0f}s]', *a, flush=True)


def sample_idx(per_class, seed):
    """the full_test rows jedi_core.sample(per_class, seed) draws (same rng calls)"""
    d = np.load(Path(os.environ['JETDISTILL_RESULTS']) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')
    y = np.asarray(d['y']); rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per_class, replace=False) for c in range(5)]))


def part_edit():
    from interp import h4_top_wmass as H
    from interp import h4_verify as HV
    from interp.h29_jedi_wpair_rebuild import p4_of, axis, model_x
    from interp.jedi_core import sample
    from interp.e42_top_wpair_jedi import dilate, clean
    TMP.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    D, meta = {}, dict(node='E42-verify', smoke=SMOKE, per_class=PER, seed=SEED, sets={}, dil={})
    x, y, _ = sample(PER, SEED)
    idx = sample_idx(PER, SEED); old = np.concatenate([sample_idx(2000, 4201), sample_idx(2000, 4202)])
    fresh = ~np.isin(idx, old); meta['overlap_with_E42_rows_removed'] = int((~fresh).sum())
    x, y = x[fresh], y[fresh]; log(f'sample {SEED}: {len(fresh)} jets, {int((~fresh).sum())} shared with E42 samples removed -> {len(y)}')
    real = x[..., 0] > 0; nn = real.sum(1).astype(np.int64)
    assert all(real[j, :nn[j]].all() for j in range(len(nn))), 'padding not contiguous'
    P4 = H._identity(p4_of(x[..., 0], x[..., 1], x[..., 2]), nn).astype(np.float32)
    log(f'identity rebuild vs stored max |d| {np.abs(model_x(P4, nn) - x).max():.2e}')
    D['y'] = y; D['U_st'] = x; D['U_rc'] = model_x(P4, nn, axis(P4, nn))
    tq = np.flatnonzero(np.isin(y, [0, 1, 4])); Pt, nt = P4[tq], nn[tq]; yb = (y[tq] == 4).astype(int); D['tq'] = tq
    O = H._obs(Pt, nt); OBS = {('nat',): O}
    win = (O['mSD'] > 105) & (O['mSD'] < 210) & (nt >= 3)
    eT = win & (yb == 1) & (O['mW'] > 65) & (O['mW'] < 95)
    d80 = np.abs(O['mW'] - 80.4); e80 = win & (yb == 1) & (d80 > 6) & (d80 < 15)
    log(f'tops {int(yb.sum())}, eligible T60/T100 {int(eT.sum())}, eligible T80 {int(e80.sum())}')
    for key in SETS:
        t = TARGET[key]; rows = np.flatnonzero(e80 if key == 'T80' else eT); t1 = time.time()
        ed = HV.edit_set(Pt, nt, rows, t); acc = {}
        for k in 'WN':
            e = ed[k]; rr = e['rows']
            if len(rr) == 0: acc[k] = set(); continue
            Oe = H._obs(e['P'], nt[rr]); dm = np.abs(Oe['m'] - O['m'][rr]); dtau = np.abs(Oe['tau32'] - O['tau32'][rr])
            okW = np.abs(Oe['mW'] - t) < TOLW[key] if k == 'W' else np.abs(Oe['mW'] - O['mW'][rr]) < 2
            acc[k] = set(rr[(dm < 0.5) & (dtau < 0.02) & okW].tolist())
        ar = np.array(sorted(acc['W'] & acc['N'] & set(ed['R']['rows'].tolist())), np.int64)
        cnt = dict(eligible=int(len(rows)), W_acc=len(acc['W']), N_acc=len(acc['N']), analysis=int(len(ar)), seconds=round(time.time() - t1))
        meta['sets'][key] = dict(counts=cnt); log(f'{key}: {cnt}')
        if len(ar) < 5: continue
        n0 = nt[ar]; D[f'rows__{key}'] = ar; D[f'mW0__{key}'] = O['mW'][ar]
        for k in 'WN':
            D[f'h__{key}__{k}'] = ed[k]['h'][np.searchsorted(ed[k]['rows'], ar)]
        Vv = {'id': Pt[ar]}
        for k in 'WNR': Vv[k] = ed[k]['P'][np.searchsorted(ed[k]['rows'], ar)]
        med = {}
        for v, P in Vv.items():
            D[f'X__{key}__{v}'] = model_x(P, n0, axis(P, n0)); Ov = H._obs(P, n0); OBS[(key, v)] = Ov
            for k in OBSK: D[f'O__{key}__{v}__{k}'] = Ov[k]
            med[v] = {k: float(np.median(Ov[k])) for k in ('m', 'mSD', 'tau21', 'tau32', 'mW')}
        meta['sets'][key]['median_obs'] = med
        meta['sets'][key]['median_h'] = {k: float(np.median(D[f'h__{key}__{k}'])) for k in 'WN'}
        log(f'{key} medians: {med}; h W/N {meta["sets"][key]["median_h"]}')
    rows = np.flatnonzero(eT); n0 = nt[rows]; D['X__DIL__id'] = model_x(Pt[rows], n0, axis(Pt[rows], n0))
    OBS[('DIL', 'id')] = {k: O[k][rows] for k in O}
    for lam in LAMS:
        v = f'L{lam:g}'; P = dilate(Pt[rows], n0, lam); D[f'X__DIL__{v}'] = model_x(P, n0, axis(P, n0)); OBS[('DIL', v)] = H._obs(P, n0)
    log(f'DIL on {len(rows)} eligible tops')
    # ---------------------------------------------------------------- references (train archive, as E42)
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from interp.h4_top_wmass import _auc
    import h5py
    lo = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
    V = Path(os.environ['JETDISTILL_DATA']); NREF = 2000 if SMOKE else 25000
    Xa = np.load(V / 'train_ptetaphi_f16.npy', mmap_mode='r')
    with h5py.File(V / 'train_meta.h5', 'r') as f: ya = f['label'][:].astype(np.int64)
    dev = np.asarray(np.load(V / 'splits.npz')['dev'], np.int64); isdev = np.zeros(len(ya), bool); isdev[dev] = True
    r_ = np.random.default_rng(4205)
    it = np.sort(np.r_[r_.choice(np.flatnonzero(~isdev & (ya == 4)), NREF, replace=False), r_.choice(np.flatnonzero(~isdev & np.isin(ya, [0, 1])), NREF, replace=False)])
    xr = np.asarray(Xa[it, :64], np.float32); del Xa; yr = (ya[it] == 4).astype(int); nr = (xr[..., 0] > 0).sum(1).astype(np.int64)
    Or = H._obs(H._identity(p4_of(xr[..., 0], xr[..., 1], xr[..., 2]), nr).astype(np.float32), nr)
    meta['ref'] = {}
    for nm, fs in REFF.items():
        feat = lambda Q, fs=fs: np.column_stack([Q[k] for k in fs]); Xf = feat(Or); sc = StandardScaler().fit(Xf)
        mlp = MLPClassifier(hidden_layer_sizes=(32, 32), early_stopping=True, random_state=0, max_iter=500).fit(sc.transform(Xf), yr)
        f_ = lambda Q, mlp=mlp, sc=sc, feat=feat: lo(mlp.predict_proba(sc.transform(feat(Q)))[:, 1])
        meta['ref'][nm] = dict(features=list(fs), fit_jets=int(len(yr)), val_acc=float(mlp.best_validation_score_),
                               test_auc_t_vs_qg=float(_auc(f_(O), yb)))
        log(f'reference {nm}: {meta["ref"][nm]}')
        for kv, Q in OBS.items():
            if kv[0] == 'nat': continue
            D[f'{nm}__{kv[0]}__{kv[1]}'] = f_(Q)
    np.savez_compressed(TMP / 'inputs.npz', **D)
    (TMP / 'edit.json').write_text(json.dumps(clean(meta), indent=1)); log(f'wrote {TMP}/inputs.npz, edit.json')


# ================================================================================================ part jedi
def radial(x):
    pt = x[..., 0].astype(np.float64); r = np.hypot(x[..., 1].astype(np.float64), x[..., 2].astype(np.float64)); real = pt > 0
    r = r * real; hinge = (np.maximum(r - 0.2, 0) * real)[:, 1:].sum(1)
    return dict(sum_r=r.sum(1), sum_r2=(r ** 2).sum(1), sum_ptr=(pt * r).sum(1), hinge02=hinge, logM2=np.log((pt * r ** 2).sum(1) + 1e-12))


def part_jedi():
    from interp.jedi_core import Jedi
    from interp.e42_top_wpair_jedi import score, call, Boot, perm_p, mcnemar, holm, clean
    rng = np.random.default_rng(4303)
    Z = np.load(TMP / 'inputs.npz'); meta = json.loads((TMP / 'edit.json').read_text())
    R = dict(node='E42-verify', smoke=SMOKE, plan='module docstring', edit=meta)
    J = Jedi(64, '3-feature'); y = Z['y']
    Lst = J.run(Z['U_st'])['logits']; Lrc = J.run(Z['U_rc'])['logits']
    R['unedited'] = dict(N=int(len(y)), acc_stored=float((Lst.argmax(1) == y).mean()), acc_recentred=float((Lrc.argmax(1) == y).mean()))
    log(f'unedited {R["unedited"]}')

    def scores(net):
        sc, lg = {}, {}
        for k in Z.files:
            if k.startswith('X__'):
                _, a, b = k.split('__'); L = net.run(np.asarray(Z[k], np.float32))['logits']; lg[(a, b)] = L; sc[(a, b)] = score(L)
        return sc, lg

    def kdil(sc, dref):
        s0 = sc[('DIL', 'id')]; dn = {l: sc[('DIL', f'L{l:g}')] - s0 for l in LAMS}
        pairs = list(zip(sorted(l for l in LAMS if l < 1), sorted((l for l in LAMS if l > 1), reverse=True)))
        out = {}
        for nm in REFF:
            r0 = Z[f'{nm}__DIL__id']; dr = {l: Z[f'{nm}__DIL__L{l:g}'] - r0 for l in LAMS}
            num = sum(dn[b] - dn[a] for a, b in pairs); den = sum(dr[b] - dr[a] for a, b in pairs); bt = Boot(len(num), 4399)
            an = np.concatenate([dn[l] for l in LAMS]); ar = np.concatenate([dr[l] for l in LAMS])
            ev_n = sum(dn[b] + dn[a] for a, b in pairs); ev_r = sum(dr[b] + dr[a] for a, b in pairs)
            out[nm] = dict(k_antisym=float(num.mean() / den.mean()), k_antisym_ci=bt.ratio_ci(num, den),
                           k_through_origin=float((an * ar).sum() / (ar ** 2).sum()), k_even=float(ev_n.mean() / ev_r.mean()),
                           dref_per_lambda={f'L{l:g}': float(dr[l].mean()) for l in LAMS})
        out['dnet_per_lambda'] = {f'L{l:g}': float(dn[l].mean()) for l in LAMS}
        return out

    def contrasts(sc, lg, key, K):
        s0, sW, sN, sR = (sc[(key, v)] for v in ('id', 'W', 'N', 'R')); n = len(s0); bt = Boot(n, 4390 + n)
        dW, dN, dR = sW - s0, sN - s0, sR - s0; x = dW - dN
        c0, cW, cN = (call(lg[(key, v)]) for v in ('id', 'W', 'N'))
        M = dict(n=n, dW=float(dW.mean()), dW_ci=bt.ci(dW), dN=float(dN.mean()), dN_ci=bt.ci(dN), dR=float(dR.mean()), dR_ci=bt.ci(dR),
                 W_minus_N=float(x.mean()), W_minus_N_ci=bt.ci(x), p_perm_WN=perm_p(x, rng), W_minus_R=float((dW - dR).mean()),
                 W_minus_R_ci=bt.ci(dW - dR), calls=dict(id=float(c0.mean()), W=float(cW.mean()), N=float(cN.mean()), mcnemar_W_vs_N=mcnemar(cW, cN)), refs={})
        for nm in REFF:
            rW = Z[f'{nm}__{key}__W'] - Z[f'{nm}__{key}__id']; rN = Z[f'{nm}__{key}__N'] - Z[f'{nm}__{key}__id']; d = rW - rN
            q = dict(W_minus_N=float(d.mean()), W_minus_N_ci=bt.ci(d), dW=float(rW.mean()), dN=float(rN.mean()))
            if K is not None:
                for kk in ('k_antisym', 'k_through_origin', 'k_even'):
                    k = K[nm][kk]; q[f'pred_net_W_minus_N__{kk}'] = float(k * d.mean())
                    q[f'ratio_net_over_pred__{kk}'] = float(x.mean() / (k * d.mean())) if abs(d.mean()) > 1e-9 else None
                q['ratio_ci__k_antisym'] = bt.ratio_ci(x, K[nm]['k_antisym'] * d)
            M['refs'][nm] = q
        return M

    sc, lg = scores(J); K = kdil(sc, None); R['k_dil'] = K; log(f'k_DIL {json.dumps(clean(K))}')
    R['sets'] = {}
    for key in SETS:
        if (key, 'W') not in sc: continue
        M = R['sets'][key] = contrasts(sc, lg, key, K)
        log(f'{key} n {M["n"]}: dW {M["dW"]:+.3f} dN {M["dN"]:+.3f} dR {M["dR"]:+.3f} W-N {M["W_minus_N"]:+.3f} {np.round(M["W_minus_N_ci"], 3)} '
            f'p {M["p_perm_WN"]:.3g}; calls {M["calls"]}; refs ' + '; '.join(f'{nm} W-N {q["W_minus_N"]:+.3f} ratio(antisym) {q.get("ratio_net_over_pred__k_antisym")}' for nm, q in M['refs'].items()))
    hp = holm([R['sets'][k]['p_perm_WN'] for k in R['sets']])
    for k, h in zip(R['sets'], hp): R['sets'][k]['p_holm_WN_3sets'] = float(h)
    # -------------------------------------------------- side-effect regression
    R['regression'] = {}
    for key in R['sets']:
        rad = {v: radial(np.asarray(Z[f'X__{key}__{v}'], np.float64)) for v in ('id', 'W', 'N')}
        x = (sc[(key, 'W')] - sc[(key, 'id')]) - (sc[(key, 'N')] - sc[(key, 'id')])
        cov = {k: rad['W'][k] - rad['N'][k] for k in rad['id']}
        for k in ('tau21', 'tau1', 'mSD'): cov[k] = Z[f'O__{key}__W__{k}'] - Z[f'O__{key}__N__{k}']
        names = list(cov); C = np.column_stack([cov[k] for k in names]); sd = C.std(0); sd[sd < 1e-12] = 1; Cs = C / sd
        A = np.column_stack([np.ones(len(x)), Cs]); beta, *_ = np.linalg.lstsq(A, x, rcond=None)
        res = x - A @ beta; r2 = 1 - res.var() / x.var()
        bi = np.random.default_rng(4310); ints = []
        for _ in range(2000):
            ii = bi.integers(0, len(x), len(x)); b_, *_ = np.linalg.lstsq(A[ii], x[ii], rcond=None); ints.append(b_[0])
        R['regression'][key] = dict(n=int(len(x)), covariates=names, intercept=float(beta[0]), intercept_ci=[float(np.quantile(ints, .025)), float(np.quantile(ints, .975))],
                                    raw_mean=float(x.mean()), r2=float(r2), coef_std=dict(zip(names, map(float, beta[1:]))),
                                    cov_mean_W_minus_N={k: float(cov[k].mean()) for k in names}, cov_sd={k: float(s) for k, s in zip(names, sd)})
        R['regression'][key]['h_median_W_N'] = [float(np.median(Z[f'h__{key}__W'])), float(np.median(Z[f'h__{key}__N'])) ]
        log(f'regression {key}: {R["regression"][key]}')
    # -------------------------------------------------- z read and S holds (T60)
    R['z_read'] = {}; R['holds'] = {}
    for key in ('T60', 'T80', 'T100'):
        if (key, 'W') not in sc: continue
        xs = {v: np.asarray(Z[f'X__{key}__{v}'], np.float32) for v in ('id', 'W', 'N')}
        st = {v: J.run(xs[v], keep=('S', 'z', 'logits')) for v in xs}; bt = Boot(len(xs['id']), 4320)
        for nm, u in (('z0', 0), ('z10', 10), ('z14', 14), ('z7', 7)):
            dWz = st['W']['z'][:, u] - st['id']['z'][:, u]; dNz = st['N']['z'][:, u] - st['id']['z'][:, u]; d = dWz - dNz
            R['z_read'][f'{key}|{nm}'] = dict(dW=float(dWz.mean()), dN=float(dNz.mean()), W_minus_N=float(d.mean()), W_minus_N_ci=bt.ci(d), p_perm=perm_p(d, rng))
        s = {v: score(st[v]['logits']) for v in xs}; resp = s['W'] - s['id']; wn = s['W'] - s['N']; hold = {}
        for u in (58, 60, 13):
            m = np.zeros(64, np.float32); m[u] = 1
            hW = score(J.run_jetwise(xs['W'], {'S': (m, st['id']['S'])})['logits']); hN = score(J.run_jetwise(xs['N'], {'S': (m, st['id']['S'])})['logits'])
            rW = resp - (hW - s['id']); rWN = wn - (hW - hN)
            hold[f'S{u}'] = dict(share_W=float(rW.sum() / resp.sum()), share_W_ci=bt.ratio_ci(rW, resp), share_WN=float(rWN.sum() / wn.sum()),
                                 WN_after_hold=float((hW - hN).mean()), WN_after_hold_ci=bt.ci(hW - hN))
        R['holds'][key] = dict(resp_W=float(resp.mean()), W_minus_N=float(wn.mean()), units=hold); log(f'holds {key}: {R["holds"][key]}')
    zk = list(R['z_read']); zp = holm([R['z_read'][k]['p_perm'] for k in zk])
    for k, h in zip(zk, zp): R['z_read'][k]['p_holm'] = float(h)
    log(f'z read {json.dumps(clean(R["z_read"]))}')
    # -------------------------------------------------- untrained
    R['untrained'] = {}
    for s_ in range(2 if SMOKE else 3):
        Ju = Jedi(64, '3-feature', untrained=True, seed=s_); scu, lgu = scores(Ju)
        R['untrained'][s_] = {k: dict(dW=float((scu[(k, 'W')] - scu[(k, 'id')]).mean()), W_minus_N=float((scu[(k, 'W')] - scu[(k, 'N')]).mean()),
                                      sd_score=float(scu[(k, 'id')].std())) for k in SETS if (k, 'W') in scu}
        log(f'untrained {s_}: {R["untrained"][s_]}'); del Ju
    (OUT / f'{TAG}.json').write_text(json.dumps(clean(R), indent=1)); log(f'wrote {OUT}/{TAG}.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--part', required=True, choices=['edit', 'jedi'])
    a = ap.parse_args(); rc = 0
    try:
        part_edit() if a.part == 'edit' else part_jedi()
    except BaseException:  # noqa: BLE001
        import traceback; traceback.print_exc(); rc = 1
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc)
