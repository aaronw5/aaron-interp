"""H19 verify, Top Landscape sample: does the fixed-axis build (top_base of the UNEDITED jet) also produce H19's low-side (T60)
result for the JetClass kin networks on Top Landscape jets?

Paths for ParT_kin / PN_kin (JetClass-trained):
  st  as H19: inputs about the constituent-sum axis of the unedited jet (top_base(P0)), also for the edited variants.
  rc  recentred: inputs about the constituent-sum axis of the jet that is scored (top_base(Pk)).
Control TN / TW (translation only): the unedited jet translated rigidly in (eta, phi) by the axis shift of the N / W edit, st path.
Top-trained nets (ParT_top, PN_top, LNet_top) are scored as in H19 (they recentre), for comparison.
Sample: Top Landscape test tops not used by H4, H4v or H19 (rng 4591, 5,000). No fresh QCD rows remain after H19, so the
  unedited accuracy uses H19's 4,360 QCD rows (rng 4192 draw, reused) together with the fresh tops.
Edit, acceptance, sets T60 / T100 / T80: as H19 (h4_verify.edit_set). References MLPx_jc, MLPxW_jc retrained as in H19.
Metric: d_k = s_k - s_id (log-odds, s = L(Tbqq) - L(QCD)); W - N with 95 % bootstrap (2,000 over jets); sign-flip p;
  ratio to MLPx_jc; Holm over sets x nets x paths (vs MLPx_jc, one-sided in the mW-window direction); McNemar W vs N at eps_S 0.5.
Run: ./part_job.sh d-h19-verify-top 'python -m interp.h19_verify_top'   (H19VT_SMOKE=1: 300 tops, 2,000 ref jets)
Outputs: $AI_RESULTS/physics/h19_verify_top.json, h19_verify_top_perjet.npz
"""
import json
import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import h19_wpair_cross as X     # noqa: E402
from interp import h19_verify as V          # noqa: E402
from interp import h4_top_wmass as H        # noqa: E402
from interp import h4_verify as HV          # noqa: E402
from interp import data_top as dt           # noqa: E402

SMOKE = bool(os.environ.get('H19VT_SMOKE'))
RES = X.RES
TAG = 'h19_verify_top_smoke' if SMOKE else 'h19_verify_top'
NT = 300 if SMOKE else 5000
B = 500 if SMOKE else 2000
X.NREF = 2000 if SMOKE else 25000
log = X.log
TBQQ, QCD = 8, 0
SETS = ('T60', 'T100', 'T80')
DIRECTION = {'T60': -1, 'T100': -1, 'T80': 1}
JCK = ('ParT_kin', 'PN_kin')
TOPN = ('ParT_top', 'PN_top', 'LNet_top')


def rows():
    """the H19 Top exclusion (H4, H4v) plus H19's own rows; fresh tops drawn with rng 4591; H19's QCD rows reused"""
    yt = np.load(dt.DST / 'test' / 'label.npy'); rng0 = np.random.default_rng(601)
    oldT = np.concatenate([rng0.choice(np.flatnonzero(yt == c), 5000, replace=False) for c in (0, 1)])
    oldQ = np.random.default_rng(602).choice(np.flatnonzero(yt == 0), 100000, replace=False)
    excl = np.concatenate([oldT, oldQ])
    idxTv, _ = HV.fresh('test', 5000, 611, (0, 1), excl); idxQv, _ = HV.fresh('test', 90000, 612, (0,), np.concatenate([excl, idxTv]))
    ok = np.ones(len(yt), bool); ok[np.concatenate([excl, idxTv, idxQv])] = False
    rng = np.random.default_rng(4192); h19 = {}
    for c in (1, 0):
        el = np.flatnonzero((yt == c) & ok); h19[c] = rng.choice(el, min(5000, len(el)), replace=False)
    ok2 = ok.copy(); ok2[np.concatenate([h19[1], h19[0]])] = False
    elT = np.flatnonzero((yt == 1) & ok2); newT = np.random.default_rng(4591).choice(elT, NT, replace=False)
    qcd = h19[0] if not SMOKE else h19[0][:200]
    info = dict(fresh_tops_available=int(len(elT)), tops_drawn=int(NT), tops_overlap_with_h19=int(len(np.intersect1d(newT, h19[1]))),
                qcd_reused_from_h19=int(len(qcd)), seed_tops=4591)
    return np.sort(np.concatenate([newT, qcd])), yt, info


def kin_scores(M, base):
    r = {}
    for name in JCK:
        L = M.jc_logits(name, X.jc_dict(base, base['_P4'], base['_nn'])); r[name] = L[:, TBQQ] - L[:, QCD]
    return r


def with_P(base, P4, nn):
    b = dict(base); b['_P4'] = P4; b['_nn'] = nn; return b


def score(M, P4, nn, base_st, order):
    """st path (fixed base axis) for the kin nets + Top nets; rc path (axis of P4) for the kin nets"""
    s = {}
    st = kin_scores(M, with_P(base_st, P4, nn)); rc = kin_scores(M, with_P(X.top_base(P4, nn), P4, nn))
    for k in JCK: s[f'{k}|st'] = st[k]; s[f'{k}|rc'] = rc[k]
    for name in TOPN:
        net = M.top[name]; s[f'{name}|st'] = net.score(P4, nn, order) if name == 'LNet_top' else net.score(P4, nn)
    return s


def main():
    RES.mkdir(parents=True, exist_ok=True)
    z = np.array([[1.0, 0, 5, 6], [0, 1.0, 2, 3], [0.5, 0.5, 1, 2]]); H._tau32(z[:, 0].copy(), z[:, 1].copy(), z[:, 2].copy(), z[:, 3].copy(), 3)
    log(f'H19 verify (Top sample) start; smoke={SMOKE}')
    idx, yt, info = rows(); log(f'rows: {info}')
    full = H._full_test(); jet = np.load(dt.DST / 'test' / 'jet.npy')
    nn = np.minimum(jet[idx, 5].astype(np.int64), full.shape[1]); P4 = H._identity(full[idx][..., :4].astype(np.float32), nn).astype(np.float32)
    y = yt[idx].astype(int); del full
    Xr, Or, yr = X.ref_train_jc(); REF = {'jc': X.fit_refs(Xr, Or, yr, 'jc')[0]}; del Xr, Or
    M = X.Models()
    O = H._obs(P4, nn); log(f'observables done on {len(nn)} jets ({int(y.sum())} top)')
    base_all = X.top_base(P4, nn); order = np.random.default_rng(605).permutation(len(nn))
    s0 = score(M, P4, nn, base_all, order); perf = {}
    for k, v in s0.items():
        perf[k] = dict(AUC=float(H._auc(v, y)), acc_binary=float(((v > 0) == (y == 1)).mean()), thr_eS50=float(np.quantile(v[y == 1], 0.5)))
    log('unedited ' + json.dumps(perf))
    win = (O['mSD'] > 105) & (O['mSD'] < 210) & (nn >= 3)
    elig = {'T60': win & (y == 1) & (O['mW'] > 65) & (O['mW'] < 95)}; elig['T100'] = elig['T60']
    d80 = np.abs(O['mW'] - H.MWREF); elig['T80'] = win & (y == 1) & (d80 > 6) & (d80 < 15)
    tgt = {'T60': (60.0, 5.0), 'T100': (100.0, 5.0), 'T80': (H.MWREF, 2.0)}
    R = dict(node='H19_verify_top', smoke=SMOKE, sample=info, unedited=perf, sets={}); PJ = {}; tests = []
    for key in SETS:
        t, tol = tgt[key]; rws = np.flatnonzero(elig[key]); ed = HV.edit_set(P4, nn, rws, t); acc = {}
        for k in 'WN':
            e = ed[k]; rr = e['rows']; Oe = H._obs(e['P'], nn[rr])
            dm = np.abs(Oe['m'] - O['m'][rr]); dtau = np.abs(Oe['tau32'] - O['tau32'][rr])
            okW = np.abs(Oe['mW'] - t) < tol if k == 'W' else np.abs(Oe['mW'] - O['mW'][rr]) < 2
            acc[k] = set(rr[(dm < 0.5) & (dtau < 0.02) & okW].tolist())
        ar = np.array(sorted(acc['W'] & acc['N']), np.int64); n = len(ar)
        cnt = dict(eligible=int(len(rws)), W_acc=len(acc['W']), N_acc=len(acc['N']), analysis=int(n)); log(f'{key} edited: {cnt}')
        if n < 5: R['sets'][key] = dict(counts=cnt); continue
        pos = {k: np.searchsorted(ed[k]['rows'], ar) for k in 'WNR'}
        P0 = P4[ar]; n0 = nn[ar]; base0 = X.top_base(P0, n0); ordr = np.random.default_rng(606).permutation(n)
        S = {'id': score(M, P0, n0, base0, ordr)}; S['id'].update(X.all_refs(REF, *H._features(P0, n0)))
        ea, pa = V.axis(P0, n0); shift = {}
        for k in 'WNR':
            Pk = ed[k]['P'][pos[k]]; S[k] = score(M, Pk, n0, base0, ordr); S[k].update(X.all_refs(REF, *H._features(Pk, n0)))
            ek, pk = V.axis(Pk, n0); shift[k] = (ek - ea, X.wrap(pk - pa))
        for k in 'WN':
            S[f'T{k}'] = score(M, V.translate(P0, n0, *shift[k]), n0, base0, ordr)
        for v, s in S.items():
            for m, a in s.items(): PJ[f'{key}|{v}|{m}'] = np.asarray(a, np.float32)
        r = dict(counts=cnt, axis_shift={k: dict(median=float(np.median(np.hypot(*shift[k]))), p90=float(np.quantile(np.hypot(*shift[k]), 0.9)))
                                         for k in 'WNR'}, models={})
        rng = np.random.default_rng(7); bi = rng.integers(0, n, (B, n)); sg = DIRECTION[key]
        d = lambda m, v: S[v][m] - S['id'][m]
        WN_ref = {ref: d(f'{ref}_jc', 'W') - d(f'{ref}_jc', 'N') for ref in ('MLPx', 'MLPxW', 'LR2')}
        for ref, x in WN_ref.items(): r['models'][f'{ref}_jc'] = dict(W_minus_N=float(x.mean()), ci95=V.ci(x, bi))
        for m in [f'{k}|{p}' for k in JCK for p in ('st', 'rc')] + [f'{k}|st' for k in TOPN]:
            dW, dN, dR = d(m, 'W'), d(m, 'N'), d(m, 'R'); x = dW - dN
            Mm = dict(dW=float(dW.mean()), dW_ci95=V.ci(dW, bi), dN=float(dN.mean()), dN_ci95=V.ci(dN, bi), dR=float(dR.mean()), dR_ci95=V.ci(dR, bi),
                      W_minus_N=float(x.mean()), ci95=V.ci(x, bi), p_perm_WN=X.perm_p(x, sg, rng),
                      dTW=float(d(m, 'TW').mean()), dTW_ci95=V.ci(d(m, 'TW'), bi), dTN=float(d(m, 'TN').mean()), dTN_ci95=V.ci(d(m, 'TN'), bi))
            for ref in ('MLPx', 'MLPxW'):
                rx = WN_ref[ref]; Mm[f'ratio_{ref}_jc'] = float(x.mean() / rx.mean()); Mm[f'ratio_{ref}_jc_ci95'] = np.percentile(x[bi].mean(1) / rx[bi].mean(1), [2.5, 97.5]).tolist()
            Mm['p_perm_vs_MLPx'] = X.perm_p(x - WN_ref['MLPx'], sg, rng); tests.append((key, m))
            thr = perf[m]['thr_eS50']; cW = S['W'][m] >= thr; cN = S['N'][m] >= thr
            Mm['calls_eS50'] = dict(pass_id=float((S['id'][m] >= thr).mean()), pass_W=float(cW.mean()), pass_N=float(cN.mean()), mcnemar_W_vs_N=H._mcnemar(cN, cW))
            if m.endswith('|st') and m.split('|')[0] in JCK:
                mr = m.replace('|st', '|rc'); dNr = d(mr, 'N'); xr = d(mr, 'W') - dNr
                Mm['dN_st_minus_rc'] = float((dN - dNr).mean()); Mm['dN_st_minus_rc_ci95'] = V.ci(dN - dNr, bi)
                Mm['p_dN_st_eq_rc_two_sided'] = float(min(1.0, 2 * min(X.perm_p(dN - dNr, 1, rng), X.perm_p(dN - dNr, -1, rng))))
                Mm['WN_st_minus_rc'] = float((x - xr).mean()); Mm['WN_st_minus_rc_ci95'] = V.ci(x - xr, bi)
            r['models'][m] = Mm
        R['sets'][key] = r; log(f'{key} scored on {n} jets')
    adj = H._holm([R['sets'][k]['models'][m]['p_perm_vs_MLPx'] for k, m in tests])
    for (k, m), pa in zip(tests, adj): R['sets'][k]['models'][m]['holm_vs_MLPx'] = float(pa)
    R['holm_family'] = f'{len(tests)} tests (sets x networks x paths), one-sided vs MLPx_jc in the mW-window direction'
    (RES / f'{TAG}.json').write_text(json.dumps(R, indent=1, default=float)); np.savez_compressed(RES / f'{TAG}_perjet.npz', **PJ)
    log(f'wrote {TAG}.json')
    for key, r in R['sets'].items():
        log(f"== {key} {r['counts']} axis shift {r.get('axis_shift')}")
        for m, Mm in r.get('models', {}).items():
            if 'dW' not in Mm: log(f"   {m:14s} W-N {Mm['W_minus_N']:+.3f} {np.round(Mm['ci95'], 3)}"); continue
            log(f"   {m:14s} dW {Mm['dW']:+.3f} dN {Mm['dN']:+.3f} W-N {Mm['W_minus_N']:+.3f} {np.round(Mm['ci95'], 3)} xMLPx_jc {Mm['ratio_MLPx_jc']:.2f} "
                f"{np.round(Mm['ratio_MLPx_jc_ci95'], 2)} holm {Mm['holm_vs_MLPx']:.3g} dTW {Mm['dTW']:+.3f} dTN {Mm['dTN']:+.3f}")


if __name__ == '__main__':
    main()
