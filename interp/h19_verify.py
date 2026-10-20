"""H19 verification (loop D): rerun the H19 W-pair edit on fresh JetClass top jets and test whether the large non-W control
response (dN) of the JetClass networks comes from the jet axis that H19 holds fixed.

Concern. H19 builds every edited JetClass jet about the STORED jet axis (interp.h19_wpair_cross.jc_dict), so a subjet move that
shifts the constituent-sum axis also translates all axis-relative inputs (deta, dphi, deltaR) of the JetClass nets. The
Top-trained nets (h4_top_wmass.jet_inputs) recompute the axis from the edited constituents. JetClass ParT/PN respond to
translated inputs (D4, D4m). The W edit and the N control move the axis by different amounts, so W - N for the JetClass
nets can contain an axis term that the Top nets do not see.
Sample: fresh JetClass test rows of TTBar_100 / ZJetsToNuNu_100 in the H19 window (550 < jet_pt < 650 GeV, |jet_eta| < 2,
  >= 3 particles), excluding the H19 rows (rng 4191 / 4192 draws of 5,000): 5,000 Tbqq (rng 4391), 3,000 QCD (rng 4392).
Edit: interp.h4_verify.edit_set unchanged, sets T60, T100, T80 and the H19 acceptance (code copied from h19 run_sample).
Paths for the JetClass nets (score L(Tbqq) - L(QCD)):
  st   H19 path: inputs about the stored jet axis (unchanged by the edit).
  rc   recentred: inputs about the constituent-sum axis of the jet that is scored (id, W, N, R each recentred).
Control T (translation only): the unedited jet translated rigidly in (eta, phi) by the axis shift that the W (TW) or N (TN)
  edit produces, scored on the st path. It measures the response to the axis shift alone.
Top nets (ParT_top, PN_top, LNet_top, PFN, EFN) are scored as in H19 (they recentre).
References: MLPx_jc / MLPxW_jc retrained as in H19 (TTBar_101 / ZJetsToNuNu_101, 25,000 per class, rng 4194 / 4193).
Statistics: bootstrap 95 % CI over jets (B = 2000), sign-flip permutation p, Holm over the vs-MLPx tests.
Run: ./part_job.sh d-h19-verify 'python -m interp.h19_verify'   (H19V_SMOKE=1: 300 + 200 jets, 2,000 ref jets)
Outputs: $AI_RESULTS/physics/h19_verify.json, h19_verify_perjet.npz
"""
import json
import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import h19_wpair_cross as X     # noqa: E402
from interp import h4_top_wmass as H        # noqa: E402
from interp import h4_verify as HV          # noqa: E402

SMOKE = bool(os.environ.get('H19V_SMOKE'))
RES = X.RES
TAG = 'h19_verify_smoke' if SMOKE else 'h19_verify'
NT, NQ = (300, 200) if SMOKE else (5000, 3000)
B = 500 if SMOKE else 2000
X.NREF = 2000 if SMOKE else 25000
log = X.log
TBQQ, QCD = 8, 0
SETS = ('T60', 'T100', 'T80')
DIRECTION = {'T60': -1, 'T100': -1, 'T80': 1}


def fresh_jc(c, k, per, seed, old_seed, old_n=5000):
    """per random rows in the H19 window, excluding the rows H19 drew with old_seed (same rng call as jc_read)"""
    import uproot
    from interp.part_core import read_root, cat, sub
    f = X.jc_file(c, k); t = uproot.open(f)['tree']
    a = t.arrays(['jet_pt', 'jet_eta', 'jet_nparticles'], library='np')
    ok = (a['jet_pt'] > 550) & (a['jet_pt'] < 650) & (np.abs(a['jet_eta']) < 2) & (a['jet_nparticles'] >= 3)
    el = np.flatnonzero(ok); old = np.random.default_rng(old_seed).choice(el, min(old_n, len(el)), replace=False)
    el2 = np.setdiff1d(el, old); idx = np.sort(np.random.default_rng(seed).choice(el2, per, replace=False)); parts = []; N = t.num_entries
    for s in range(0, N, 20000):
        loc = idx[(idx >= s) & (idx < s + 20000)] - s
        if len(loc): parts.append(sub(read_root(f, start=s, stop=min(s + 20000, N)), loc))
    info = dict(file=f.name, eligible=int(len(el)), excluded_h19=int(len(old)), drawn=int(len(idx)), overlap_with_h19=int(len(np.intersect1d(idx, old))), seed=seed)
    log(f'fresh {c}: {info}')
    return cat(parts), idx, info


def axis(P4, nn):
    P = P4.astype(np.float64); real = np.arange(P.shape[1])[None, :] < nn[:, None]
    S = (P * real[..., None]).sum(1); pt = np.hypot(S[:, 0], S[:, 1])
    return np.arcsinh(S[:, 2] / pt), np.arctan2(S[:, 1], S[:, 0])


def jc_dict_rc(base, P4, nn):
    e, p = axis(P4, nn); b = dict(base); b['jet_eta'] = e.astype(np.float32); b['jet_phi'] = p.astype(np.float32)
    return X.jc_dict(b, P4, nn)


def translate(P4, nn, de, dp):
    """every constituent moved by (de, dp) in (eta, phi); pT and mass kept"""
    P = P4.astype(np.float64); real = np.arange(P.shape[1])[None, :] < nn[:, None]
    pt = np.hypot(P[..., 0], P[..., 1]); ptc = np.where(real, pt, 1.0)
    eta = np.arcsinh(np.where(real, P[..., 2], 0) / ptc) + de[:, None]; phi = np.arctan2(P[..., 1], P[..., 0]) + dp[:, None]
    m = np.sqrt(np.maximum(P[..., 3] ** 2 - (P[..., :3] ** 2).sum(-1), 0))
    out = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2)], -1)
    return (out * real[..., None]).astype(np.float32)


def jc_scores(M, Jd):
    r, lg = {}, {}
    for name in M.jc:
        L = M.jc_logits(name, Jd); r[name] = L[:, TBQQ] - L[:, QCD]; lg[name] = L
    return r, lg


def ci(x, bi):
    return np.percentile(x[bi].mean(1), [2.5, 97.5]).tolist()


def main():
    RES.mkdir(parents=True, exist_ok=True)
    z = np.array([[1.0, 0, 5, 6], [0, 1.0, 2, 3], [0.5, 0.5, 1, 2]]); H._tau32(z[:, 0].copy(), z[:, 1].copy(), z[:, 2].copy(), z[:, 3].copy(), 3)
    log(f'H19 verify start; smoke={SMOKE}')
    from interp.part_core import cat, kin
    Jt, _, it = fresh_jc('Tbqq', 100, NT, 4391, 4191); Jq, _, iq = fresh_jc('QCD', 100, NQ, 4392, 4192)
    J = cat([Jt, Jq]); P4raw, nn = X.jc_P4(J); P4 = H._identity(P4raw, nn).astype(np.float32); y = (J['label'] == TBQQ).astype(int)
    J['pmass'] = kin(J)[3].astype(np.float32)
    Xr, Or, yr = X.ref_train_jc(); REF = {'jc': X.fit_refs(Xr, Or, yr, 'jc')[0]}; del Xr, Or
    M = X.Models(); M.massless = True   # Top Landscape constituents are massless (H19: 0.13 % above 0.1 GeV)
    O = H._obs(P4, nn); log('observables done')
    # ---------------------------------------------------------------- unedited, both paths
    order = np.random.default_rng(605).permutation(len(nn))
    s_st, lg_st = M.score('jc', P4, nn, J, order, want_logits=True); s_rc, lg_rc = jc_scores(M, jc_dict_rc(J, P4, nn))
    e0, p0 = axis(P4, nn); dax = np.hypot(e0 - J['jet_eta'], X.wrap(p0 - J['jet_phi']))
    perf = {}
    for path, s, lg in (('st', s_st, lg_st), ('rc', s_rc, lg_rc)):
        for k, v in s.items():
            if path == 'rc' and k not in M.jc: continue
            d = dict(AUC=float(H._auc(v, y)), acc_binary=float(((v > 0) == (y == 1)).mean()))
            if k in lg: d['acc_10class'] = float((lg[k].argmax(1) == np.where(y == 1, TBQQ, QCD)).mean())
            d['thr_eS50'] = float(np.quantile(v[y == 1], 0.5)); perf[f'{k}|{path}'] = d
    log('unedited ' + json.dumps(perf))
    log(f'stored vs constituent-sum axis on unedited jets: median dR {np.median(dax):.2e}, p99 {np.quantile(dax, 0.99):.2e}, max {dax.max():.2e}')
    # ---------------------------------------------------------------- edits (H19 acceptance)
    win = (O['mSD'] > 105) & (O['mSD'] < 210) & (nn >= 3)
    elig = {'T60': win & (y == 1) & (O['mW'] > 65) & (O['mW'] < 95)}; elig['T100'] = elig['T60']
    d80 = np.abs(O['mW'] - H.MWREF); elig['T80'] = win & (y == 1) & (d80 > 6) & (d80 < 15)
    tgt = {'T60': (60.0, 5.0), 'T100': (100.0, 5.0), 'T80': (H.MWREF, 2.0)}
    R = dict(node='H19_verify', smoke=SMOKE, samples=dict(Tbqq=it, QCD=iq), unedited=perf,
             axis_stored_vs_sum=dict(median=float(np.median(dax)), p99=float(np.quantile(dax, 0.99))), sets={})
    PJ = {}; tests = []
    for key in SETS:
        t, tol = tgt[key]; rows = np.flatnonzero(elig[key]); ed = HV.edit_set(P4, nn, rows, t); acc = {}
        for k in 'WN':
            e = ed[k]; rr = e['rows']; Oe = H._obs(e['P'], nn[rr])
            dm = np.abs(Oe['m'] - O['m'][rr]); dtau = np.abs(Oe['tau32'] - O['tau32'][rr])
            okW = np.abs(Oe['mW'] - t) < tol if k == 'W' else np.abs(Oe['mW'] - O['mW'][rr]) < 2
            acc[k] = set(rr[(dm < 0.5) & (dtau < 0.02) & okW].tolist())
        ar = np.array(sorted(acc['W'] & acc['N']), np.int64); n = len(ar)
        cnt = dict(eligible=int(len(rows)), W_acc=len(acc['W']), N_acc=len(acc['N']), analysis=int(n)); log(f'{key} edited: {cnt}')
        if n < 5: R['sets'][key] = dict(counts=cnt); continue
        pos = {k: np.searchsorted(ed[k]['rows'], ar) for k in 'WNR'}
        base = X.subd(J, ar); ordr = np.random.default_rng(606).permutation(n); P0 = P4[ar]; n0 = nn[ar]
        S = {'id|st': M.score('jc', P0, n0, base, ordr), 'id|rc': jc_scores(M, jc_dict_rc(base, P0, n0))[0]}
        S['id|st'].update(X.all_refs(REF, *H._features(P0, n0)))
        ea, pa = axis(P0, n0); shift = {}
        for k in 'WNR':
            Pk = ed[k]['P'][pos[k]]
            S[f'{k}|st'] = M.score('jc', Pk, n0, base, ordr); S[f'{k}|st'].update(X.all_refs(REF, *H._features(Pk, n0)))
            S[f'{k}|rc'] = jc_scores(M, jc_dict_rc(base, Pk, n0))[0]
            ek, pk = axis(Pk, n0); shift[k] = (ek - ea, X.wrap(pk - pa))
        for k in 'WN':   # translation-only control: the unedited jet moved by the axis shift of edit k
            S[f'T{k}|st'] = M.score('jc', translate(P0, n0, *shift[k]), n0, base, ordr)
        for v, s in S.items():
            for m, a in s.items(): PJ[f'{key}|{v}|{m}'] = np.asarray(a, np.float32)
        r = dict(counts=cnt, axis_shift={k: dict(median=float(np.median(np.hypot(*shift[k]))), p90=float(np.quantile(np.hypot(*shift[k]), 0.9)))
                                         for k in 'WNR'}, models={})
        rng = np.random.default_rng(7); bi = rng.integers(0, n, (B, n)); sg = DIRECTION[key]
        d = lambda m, v, p: S[f'{v}|{p}'][m] - S[f'id|{p}'][m]
        WN_ref = {ref: d(f'{ref}_jc', 'W', 'st') - d(f'{ref}_jc', 'N', 'st') for ref in ('MLPx', 'MLPxW', 'LR2')}
        for ref, x in WN_ref.items(): r['models'][f'{ref}_jc'] = dict(W_minus_N=float(x.mean()), ci95=ci(x, bi))
        for m in list(M.jc) + list(M.top) + list(M.flows):
            for p in (('st', 'rc') if m in M.jc else ('st',)):
                dW, dN, dR = d(m, 'W', p), d(m, 'N', p), d(m, 'R', p); x = dW - dN
                Mm = dict(dW=float(dW.mean()), dW_ci95=ci(dW, bi), dN=float(dN.mean()), dN_ci95=ci(dN, bi), dR=float(dR.mean()), dR_ci95=ci(dR, bi),
                          W_minus_N=float(x.mean()), ci95=ci(x, bi), p_perm_WN=X.perm_p(x, sg, rng))
                for ref in ('MLPx', 'MLPxW'):
                    rx = WN_ref[ref]; bm = rx[bi].mean(1)
                    Mm[f'ratio_{ref}'] = float(x.mean() / rx.mean()); Mm[f'ratio_{ref}_ci95'] = np.percentile(x[bi].mean(1) / bm, [2.5, 97.5]).tolist()
                Mm['p_perm_vs_MLPx'] = X.perm_p(x - WN_ref['MLPx'], sg, rng); tests.append((key, f'{m}|{p}'))
                thr = perf[f'{m}|{p}']['thr_eS50']; c0 = S[f'id|{p}'][m] >= thr; cW = S[f'W|{p}'][m] >= thr; cN = S[f'N|{p}'][m] >= thr
                Mm['calls_eS50'] = dict(pass_id=float(c0.mean()), pass_W=float(cW.mean()), pass_N=float(cN.mean()), mcnemar_W_vs_N=H._mcnemar(cN, cW))
                if p == 'st':
                    dTW, dTN = d(m, 'TW', 'st'), d(m, 'TN', 'st')
                    Mm.update(dTW=float(dTW.mean()), dTW_ci95=ci(dTW, bi), dTN=float(dTN.mean()), dTN_ci95=ci(dTN, bi))
                    if m in M.jc:   # paired change of dN and of W - N between the stored-axis and recentred paths
                        dNr = d(m, 'N', 'rc'); xr = d(m, 'W', 'rc') - dNr
                        Mm['dN_st_minus_rc'] = float((dN - dNr).mean()); Mm['dN_st_minus_rc_ci95'] = ci(dN - dNr, bi)
                        Mm['p_dN_st_eq_rc_two_sided'] = float(min(1.0, 2 * min(X.perm_p(dN - dNr, 1, rng), X.perm_p(dN - dNr, -1, rng))))
                        Mm['WN_st_minus_rc'] = float((x - xr).mean()); Mm['WN_st_minus_rc_ci95'] = ci(x - xr, bi)
                r['models'][f'{m}|{p}'] = Mm
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
            log(f"   {m:14s} dW {Mm['dW']:+.3f} dN {Mm['dN']:+.3f} {np.round(Mm['dN_ci95'], 2)} dR {Mm['dR']:+.3f} W-N {Mm['W_minus_N']:+.3f} {np.round(Mm['ci95'], 3)} "
                f"xMLPx {Mm['ratio_MLPx']:.2f} {np.round(Mm['ratio_MLPx_ci95'], 2)} holm {Mm['holm_vs_MLPx']:.3g}"
                + (f" dTW {Mm['dTW']:+.3f} dTN {Mm['dTN']:+.3f}" if 'dTW' in Mm else '')
                + (f" dN st-rc {Mm['dN_st_minus_rc']:+.3f} {np.round(Mm['dN_st_minus_rc_ci95'], 3)}" if 'dN_st_minus_rc' in Mm else ''))


if __name__ == '__main__':
    main()
