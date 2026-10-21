"""E37x verify: fresh-sample rerun of E37x's key read-outs plus three controls E37x did not run.

Network: official JEDI-linear 3-feature n64 (fixed point), exact h4 tail (x10 JNet), as E37x.
Samples: T = W+Z jets of interp.jedi_core.sample(3000, 4747) (test); F = W+Z jets of sample(3000, 4748) minus T rows (fit).
  Seeds differ from E37x (3737 / 3738); overlap of rows with E37x's T and F is reported.
Rerun (E37x code imported unchanged): DAS refit of v on F; Ez / Edr / Em / Erot (dz +-0.15, +-0.075), Emf, D(+-0.3) on T;
  calibrated slopes R_z, R_Edr, R_rot (readout_core); naming s_q; interaction model and per-bin slopes a_bin; holds under
  D(+0.3) and Ez(+0.15) (hold v, hold M2, hold v+M2, copy v+M2, copy v) with 500 random subspaces per dimension.
New controls:
  (A) direct steering of u at h4: h4' = clip(h4 + delta v), delta in {-1, -0.5, +0.5, +1} (u units; v unit norm), exact tail;
      per-jet slope dmargin/du (least squares through 0 over the four deltas); mean per unmodified-jet mass bin and class with
      2,000 jet-bootstrap CI.  Null: 200 random unit directions in the live h4 units (same deltas).  Statistic for the sign
      change: d = slope(85-95) - slope(75-85) per class, compared with the random directions' d (two-sided rank p).
      Tests the causal content of E37x's per-bin regression slopes a_bin, which pool edit types.
  (B) Ez decomposition on Ez-accepted T jets (dz = +-0.15):
      Ept  = only the prong pT rescaling of Ez (prong angular shapes kept, so prong masses scale with prong pT; Delta R12 exact).
      Ewid = only the prong-width rescaling of Ez (each prong's offsets about its own axis scaled by Ez's factor, pT kept, prong
             axes kept; z and Delta R12 exact; prong masses scale with the width factor).
      Each with a dilation Em matched to its own jet-mass change (EmP, EmW).  Shows how much of Ez's Delta u comes from the
      change of z itself and how much from the soft prong's angular widening needed to hold its mass.
  (C) dilation scan: all angles about the jet axis scaled by f in FSCAN on all T jets: mean Delta u by class (shape of u vs
      angular scale, to check E37x's Em rows, which move u up for both signs of the mass change).
Writes $AI_RESULTS/jedi/e37x_verify.json.  Run: ./job.sh e37x-verify 'python -m interp.e37x_verify'
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import e37x_sharing_physics_jedi as X  # noqa: E402

E, H50 = X.E, X.H50
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'
T0 = time.time()
FSCAN = (0.80, 0.85, 0.90, 0.95, 1.05, 1.10, 1.15, 1.20)
STEER = (-1.0, -0.5, 0.5, 1.0)


def log(*a): print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def decomp_variants(DT, ACCz):
    """Ept / Ewid / EmP / EmW for dz = +-0.15 on Ez-accepted jets; dilation scan on all jets"""
    P4, nn, LAB = DT['P4'], DT['nn'], DT['LAB']; J = len(P4)
    keys = [(k, dz) for dz in (-0.15, 0.15) for k in ('Ept', 'Ewid', 'EmP', 'EmW')] + [('Dil', f) for f in FSCAN]
    PV = {k: P4.copy() for k in keys}; ACC = {k: np.zeros(J, bool) for k in keys}; MM = {k: np.full(J, np.nan) for k in keys}
    SK = {dz: np.full((J, 2), np.nan) for dz in (-0.15, 0.15)}; m0 = np.full(J, np.nan); t = time.time()
    for j in range(J):
        n = int(nn[j])
        if n < 3: continue
        P = P4[j, :n]; lab = LAB[j, :n]; pt, y, phi, _ = H50.kin(P); m = np.zeros_like(pt)
        T = P.sum(0); ya, pa = float(X.rap(T)), float(np.arctan2(T[1], T[0])); m0[j] = H50.jm(pt, y, phi, m)
        dil = lambda f: (ya + f * (y - ya), pa + f * X.wrap(phi - pa))
        for f in FSCAN:
            Q = H50.build(pt, *dil(f), m); PV[('Dil', f)][j, :n] = Q; ACC[('Dil', f)][j] = True; MM[('Dil', f)][j] = H50.mass4(Q.sum(0))
        M0 = [H50.pmass(pt, y, phi, m, lab == k) for k in (0, 1)]
        for dz in (-0.15, 0.15):
            if not ACCz[dz][j]: continue
            r, inf = H50.arm_c(pt, y, phi, m, lab, -dz, M0)
            if r is None: continue
            pt2, y2, p2 = r
            # width factor of each prong: ratio of constituent distances to the prong axis (uniform pT scaling keeps the axis)
            sk = []
            for k in (0, 1):
                s = lab == k
                if s.sum() < 2: sk.append(1.0); continue
                a1 = H50.paxis(pt, y, phi, m, s); a2 = H50.paxis(pt2, y2, p2, m, s)
                d1 = np.hypot(y[s] - a1[0], X.wrap(phi[s] - a1[1])); d2 = np.hypot(y2[s] - a2[0], X.wrap(p2[s] - a2[1]))
                w = pt[s] * d1; sk.append(float((w * d2).sum() / max((w * d1).sum(), 1e-30)))
            SK[dz][j] = sk
            # Ept: pT rescaling only (prong axes unchanged by a uniform pT scaling within each prong)
            QP = H50.build(pt2, y, phi, m)
            # Ewid: width rescaling only, about each prong's own axis, prong axes kept
            yw, pw = y, phi
            for k in (0, 1):
                s = lab == k
                if s.sum() >= 2: yw, pw = H50.scale_prong(pt, yw, pw, m, s, sk[k])
            QW = H50.build(pt, yw, pw, m)
            for nm, Q in (('Ept', QP), ('Ewid', QW)):
                PV[(nm, dz)][j, :n] = Q; ACC[(nm, dz)][j] = True; MM[(nm, dz)][j] = H50.mass4(Q.sum(0))
                kk = H50.bisect(lambda q: H50.jm(pt, *dil(q), m), 0.2, 5.0, MM[(nm, dz)][j])
                if kk is not None:
                    key = ('EmP' if nm == 'Ept' else 'EmW', dz); Qm = H50.build(pt, *dil(kk), m)
                    PV[key][j, :n] = Qm; ACC[key][j] = True; MM[key][j] = H50.mass4(Qm.sum(0))
        if j % 1000 == 0: log(f'  decomposition variants: jet {j}/{J} [{time.time() - t:.0f}s]')
    Xd = {}
    for k in keys:   # memory: convert and drop each edited-jet array in turn
        Pk = PV.pop(k); Xd[k] = E.model_xn(Pk, nn, X.axis(Pk, nn))[0]; del Pk
    return Xd, ACC, MM, m0, SK


def steering(net, proj, h0, m0, dirs, y, mT, W, live, nrand):
    rg = net.rng_['h4']; bins = ['all'] + [f'{lo}-{hi}' for lo, hi in X.BINS]

    def per_jet(vv):
        num = np.zeros(len(h0)); den = 0.0; dms = {}
        for d in STEER:
            dm = X.margin(net.exact('h4', proj(rg, h0, h0 + d * vv[None, :]))[0]) - m0; num += d * dm; den += d * d; dms[d] = dm
        return num / den, dms

    def sel(bn, c):
        s = (y == c)
        if bn != 'all':
            lo, hi = map(float, bn.split('-')); s &= (mT >= lo) & (mT < hi)
        return np.flatnonzero(s)
    out = {}
    for nm, vv in dirs.items():
        sl, dms = per_jet(vv); o = {}
        for c, cn in ((2, 'W'), (3, 'Z')):
            for bn in bins:
                s = sel(bn, c)
                if len(s) < 10: continue
                o[f'{bn}|{cn}'] = dict(n=int(len(s)), slope=float(sl[s].mean()), ci=X.qci(X.bmean(W, sl, s)),
                                       dm_plus1=float(dms[1.0][s].mean()), dm_minus1=float(dms[-1.0][s].mean()))
            a, b = sel('75-85', c), sel('85-95', c)
            if len(a) >= 10 and len(b) >= 10:
                dd = X.bmean(W, sl, b) - X.bmean(W, sl, a)
                o[f'diff_85-95_minus_75-85|{cn}'] = dict(d=float(sl[b].mean() - sl[a].mean()), ci=X.qci(dd), p_boot=X.pboot(dd))
        out[nm] = o
    # random directions
    rr = np.random.default_rng(77); R = {f'{bn}|{cn}': [] for bn in bins for cn in 'WZ'}; RD = {'W': [], 'Z': []}
    for i in range(nrand):
        g = E.rand_live(rr, live, 64, 1)[:, 0]; sl, _ = per_jet(g)
        for c, cn in ((2, 'W'), (3, 'Z')):
            for bn in bins:
                s = sel(bn, c)
                if len(s) >= 10: R[f'{bn}|{cn}'].append(float(sl[s].mean()))
            a, b = sel('75-85', c), sel('85-95', c)
            if len(a) >= 10 and len(b) >= 10: RD[cn].append(float(sl[b].mean() - sl[a].mean()))
    out['random'] = {k: dict(n=len(v), abs_median=float(np.median(np.abs(v))), q=np.quantile(v, [.025, .5, .975]).tolist()) for k, v in R.items() if v}
    out['random_diff'] = {cn: dict(n=len(v), abs_median=float(np.median(np.abs(v))), q=np.quantile(v, [.025, .5, .975]).tolist()) for cn, v in RD.items() if v}
    for nm in dirs:
        for cn in 'WZ':
            k = f'diff_85-95_minus_75-85|{cn}'
            if k in out[nm] and RD[cn]:
                out[nm][k]['rank_p_vs_random_abs'] = float((1 + sum(abs(r) >= abs(out[nm][k]['d']) for r in RD[cn])) / (1 + len(RD[cn])))
        for k in R:
            if k in out[nm] and R[k]:
                out[nm][k]['rank_p_vs_random_abs'] = float((1 + sum(abs(r) >= abs(out[nm][k]['slope']) for r in R[k])) / (1 + len(R[k])))
    return out


def main(a):
    from interp.jedi_core import sample
    from interp.x10_das_jedi import JNet, proj
    import jax
    TS, FS, per = a.tseed, a.fseed, a.per
    B = a.boot; OUT.mkdir(parents=True, exist_ok=True); tag = 'e37x_verify' + a.suffix
    res = dict(node='E37x-verify', seeds=dict(T=TS, F=FS), per_class=per, B=B, nrand_holds=a.nrand_holds, nrand_steer=a.nrand_steer)
    save = lambda: (OUT / f'{tag}.json').write_text(json.dumps(X.jsonable(res), indent=1))
    xT, yT, _ = sample(per, TS); xF, yF, _ = sample(per, FS); rT = E.rows_of(per, TS); rF = E.rows_of(per, FS)
    xT = np.asarray(xT, np.float32); xF = np.asarray(xF, np.float32)
    rT0, rF0 = E.rows_of(3000, 3737), E.rows_of(3000, 3738)
    res['overlap_rows'] = dict(T_in_E37x_T=int(np.isin(rT, rT0).sum()), T_in_E37x_F=int(np.isin(rT, rF0).sum()), F_in_T=int(np.isin(rF, rT).sum()))
    net = JNet(); net.off = np.zeros(5)
    R0 = net.reps(xT, keep=('S', 'h4', 'h5', 'logits')); res['acc_T_5class_stored'] = float((R0['logits'].argmax(1) == yT).mean())
    res['tail_check'] = net.check(xT[:2000], {k: v[:2000] for k, v in R0.items()}); del R0
    log(f'acc T {res["acc_T_5class_stored"]:.4f}; overlap {res["overlap_rows"]}')
    DF = E.prep(xF, yF, rF, FS, exclude_rows=rT); DT = E.prep(xT, yT, rT, TS)
    J = DT['J']; y = DT['y']; res['N'] = dict(T_WZ=J, F_WZ=DF['J'])
    del xT, xF; DT.pop('x', None); DF.pop('x', None)
    RF = E.reps_all(net, DF); RT = E.reps_all(net, DT)
    DF.pop('PE', None); DF['X'] = {}   # memory: F's edited inputs are not used after the representations
    for k in [k for k in DT['PE'] if k != (0.3, 'D') and k != (-0.3, 'D')]: DT['PE'].pop(k)
    live = [int(u) for u in np.flatnonzero(RT['id']['h4'].std(0) > 1e-9)]; lm = np.isin(np.arange(64), live); res['n_live_h4'] = len(live)
    PF = E.build_pairs(DF, RF, 'h4'); fit = [PF['self'], PF['cross']]
    hb = np.concatenate([p.hb for p in fit]); hs = np.concatenate([p.hs for p in fit]); tg = np.concatenate([p.mt for p in fit]); grp = np.concatenate([p.jet for p in fit])
    Q1, info = E.das_fit_margin(net, 'h4', hb, hs, tg, grp, 1, live, seed=TS, steps=400); jax.clear_caches()
    del hb, hs, tg, grp, PF, fit
    v = Q1[:, 0].copy()
    QS = dict(np.load(OUT / 'e37_das_Q.npz')); MS = dict(np.load(OUT / 'e37_das_M.npz'))
    M2 = MS['M2_h4'].copy(); M2[~lm] = 0; M2 = np.linalg.qr(M2)[0]; v37 = QS['h4_k1'][:, 0].copy(); v37[~lm] = 0; v37 /= np.linalg.norm(v37)
    OF = np.full((DF['J'], len(X.QN)), np.nan)
    for j in range(DF['J']):
        if DF['nn'][j] >= 3: OF[j] = X.obs1(DF['P4'][j, :DF['nn'][j]], DF['LAB'][j, :DF['nn'][j]])
    okF = np.isfinite(OF).all(1)
    if np.corrcoef(RF['id']['h4'][okF] @ v, OF[okF, 0])[0, 1] < 0: v = -v
    if np.corrcoef(RF['id']['h4'][okF] @ v37, OF[okF, 0])[0, 1] < 0: v37 = -v37
    PT = E.build_pairs(DT, RT, 'h4')
    res['direction'] = dict(das_train=info, cos_v_vs_stored_E37=float(abs(v @ v37)), max_cos_v_vs_M2=float(np.linalg.norm(M2.T @ v)),
                            corr_u_z_F=float(np.corrcoef(RF['id']['h4'][okF] @ v, OF[okF, 0])[0, 1]),
                            T_interchange={nm: E.metrics(E.patch(net, 'h4', PT[nm].hb, PT[nm].hs, Q1), PT[nm]) for nm in ('self', 'dil')})
    del PT
    log('direction ' + json.dumps(X.jsonable(res['direction']))); save()
    # ---- E37x variants (unchanged code) and the new decomposition / dilation scan
    Xv, ACC, OB, OT, PAR, reasons, chk, _ = X.build_variants(DT, TS, 0)
    DT.pop('PE', None); DT['X'] = {}
    res['edits'] = dict(accepted={f'{k[0]}{k[1]:+}': float(ACC[k].mean()) for k in ACC}, max_invariant_error=chk)
    Xd, ACCd, MMd, m0j, SK = decomp_variants(DT, {dz: ACC[('Ez', dz)] for dz in (-0.15, 0.15)})
    res['edits']['width_factor'] = {f'{dz:+}': dict(hard_or_0=float(np.nanmedian(SK[dz][:, 0])), prong1=float(np.nanmedian(SK[dz][:, 1]))) for dz in SK}
    mT = OT[:, X.QN.index('m')]
    REP = {'id': dict(h4=RT['id']['h4'], logits=RT['id']['logits'])}
    for k, xv in Xv.items(): REP[k] = net.reps(xv, keep=('h4', 'logits'))
    for k, xv in Xd.items(): REP[k] = net.reps(xv, keep=('h4', 'logits'))
    kinds = list(Xv.keys()); del Xd, Xv
    m0 = X.margin(REP['id']['logits']); u0 = REP['id']['h4'] @ v
    W = X.bweights(J, B, 11); WF = X.bweights(DF['J'], B, 12)
    DU = {k: REP[k]['h4'] @ v - u0 for k in REP if k != 'id'}; DM = {k: X.margin(REP[k]['logits']) - m0 for k in REP if k != 'id'}
    # table (all bins) for the +-0.15 cells
    tab = {}
    for dz in (-0.15, 0.15):
        s0 = ACC[('Ez', dz)]
        for kk in ('Ez', 'Edr', 'Em', 'Erot', 'Ept', 'Ewid', 'EmP', 'EmW'):
            k = (kk, dz); acc = ACC.get(k, ACCd.get(k)); mm = OB[k][:, X.QN.index('m')] if k in OB else MMd[k]
            for c, cn in ((2, 'W'), (3, 'Z')):
                sel = s0 & acc & (y == c); s = np.flatnonzero(sel)
                tab[f'{kk}{dz:+}|{cn}'] = dict(n=int(len(s)), du=X.mstat(W, DU[k], sel), dmargin=X.mstat(W, DM[k], sel),
                                               dlogm=float(np.log(mm[s] / mT[s]).mean()) if len(s) else None)
        # paired decomposition contrasts on jets where all of Ez, Em, Ept, EmP, Ewid, EmW exist
        for c, cn in ((2, 'W'), (3, 'Z'), (None, 'WZ')):
            sel = s0 & ACCd[('Ept', dz)] & ACCd[('EmP', dz)] & ACCd[('Ewid', dz)] & ACCd[('EmW', dz)] & ((y == c) if c else True)
            for nm, a_, b_ in (('Ez_minus_Em', ('Ez', dz), ('Em', dz)), ('Ept_minus_EmP', ('Ept', dz), ('EmP', dz)), ('Ewid_minus_EmW', ('Ewid', dz), ('EmW', dz))):
                tab[f'contrast{dz:+}|{nm}|{cn}'] = X.mstat(W, DU[a_] - DU[b_], sel)
    res['table'] = tab
    for k_, o in tab.items():
        if 'du' in o: log(f'  {k_}: n {o["n"]} du {o["du"].get("mean")} {o["du"].get("ci")} dmargin {o["dmargin"].get("mean")} dlogm {o["dlogm"]}')
        else: log(f'  {k_}: {o.get("mean")} {o.get("ci")} n {o.get("n")}')
    scan = {}
    for f in FSCAN:
        k = ('Dil', f)
        for c, cn in ((2, 'W'), (3, 'Z')):
            sel = (y == c) & ACCd[k]; scan[f'{f}|{cn}'] = dict(du=X.mstat(W, DU[k], sel), dmargin=X.mstat(W, DM[k], sel), dlogm=float(np.log(MMd[k][sel] / mT[sel]).mean()))
    res['dilation_scan'] = scan; log('dilation scan ' + json.dumps(X.jsonable({k: (o['du'].get('mean'), o['dlogm']) for k, o in scan.items()})))
    save()
    # ---- E37x read-outs on the fresh sample
    DUo = {k: DU[k] for k in kinds}
    r = X.readout_core(RF['id']['h4'] @ v, OF, u0, OT, DUo, ACC, OB, y, W, WF, mT, 'das')
    res['readout'] = dict(beta_z=r['beta_z_knn']['all'], pooled=r['calibrated'].get('all|WZ', {}).get('pooled'),
                          per_bin={k: v_.get('pooled') for k, v_ in r['calibrated'].items() if k.endswith('|WZ')},
                          naming={q: {kk: o[kk] for kk in ('s_q', 's_q_ci', 's_q_by_edit', 'heldout_partial_R2')} for q, o in r['naming']['per_quantity'].items()},
                          closest_to_1=r['naming']['closest_to_1'], dilation_gain=r['dilation_gain_du_per_dlogm'])
    log('readout ' + json.dumps(X.jsonable(res['readout']))[:3000]); save()
    DLM = {k: np.log(OB[k][:, X.QN.index('m')] / mT) for k in kinds}
    res['interaction'] = X.interaction(DUo, {k: DM[k] for k in kinds}, DLM, ACC, y, mT, W, kinds)
    # per-bin slopes without the D rows (the D rows have the largest du at fixed mass)
    kinds_noD = [k for k in kinds if k[0] != 'D']
    res['interaction_noD'] = X.interaction(DUo, {k: DM[k] for k in kinds}, DLM, ACC, y, mT, W[:300], kinds_noD)
    # D rows alone (jet mass fixed, so no dlogm term): dmargin ~ 1 + a du per bin and class
    dsl = {}
    for c, cn in ((2, 'W'), (3, 'Z')):
        for lo, hi in X.BINS:
            du_, dm_, js_ = [], [], []
            for k in (('D', -0.3), ('D', 0.3)):
                s = np.flatnonzero(ACC[k] & (y == c) & (mT >= lo) & (mT < hi)); du_.append(DU[k][s]); dm_.append(DM[k][s]); js_.append(s)
            du_, dm_, js_ = map(np.concatenate, (du_, dm_, js_))
            if len(js_) < 30: continue
            def fit(w):
                Xm = np.stack([np.ones(len(du_)), du_], 1); sw = np.sqrt(w); return np.linalg.lstsq(Xm * sw[:, None], dm_ * sw, rcond=None)[0][1]
            bb = [fit(w[js_]) for w in W[:300].astype(np.float64)]
            dsl[f'{lo}-{hi}|{cn}'] = dict(n_rows=int(len(js_)), a_du=float(fit(np.ones(len(js_)))), a_ci=X.qci(bb))
    res['Donly_per_bin_slope'] = dsl; log('D-only per-bin slopes ' + json.dumps(X.jsonable(dsl)))
    for nm in ('interaction', 'interaction_noD'):
        log(nm + ' ' + json.dumps(X.jsonable({c: {b: (o['a_du'], o['a_ci']) for b, o in res[nm][c]['per_bin'].items()} for c in ('W', 'Z')})))
    save()
    # ---- (A) direct steering of u per mass bin
    res['steering'] = steering(net, proj, REP['id']['h4'], m0, {'v': v, 'v_E37_stored': v37}, y, mT, W, live, a.nrand_steer)
    log('steering ' + json.dumps(X.jsonable({nm: {k: (o.get('slope', o.get('d')), o.get('ci')) for k, o in res['steering'][nm].items()} for nm in ('v', 'v_E37_stored')})))
    log('steering random ' + json.dumps(X.jsonable(res['steering']['random_diff'])))
    save()
    # ---- holds
    Qv = v[:, None]; Qb = np.linalg.qr(np.c_[v, M2])[0]
    sets = {'i_hold_v': (Qv, 'hold'), 'ii_hold_M2': (M2, 'hold'), 'iii_hold_v_M2': (Qb, 'hold'), 'iv_copy_v_M2': (Qb, 'copy'), 'x_copy_v': (Qv, 'copy')}
    res['holds'] = {}
    for ek in (('D', 0.3), ('Ez', 0.15)):
        sel = ACC[ek]; o = X.holds(net, proj, REP['id']['h4'], REP[ek]['h4'], m0, X.margin(REP[ek]['logits']), y, sel, sets, live, a.nrand_holds, W, 41 if ek[0] == 'D' else 42)
        res['holds'][f'{ek[0]}{ek[1]:+}'] = o
        log(f'holds {ek}: free {o["free_dmargin"]}; ' + '; '.join(f'{nm} {o[nm]["share"]} {o[nm]["ci"]} p {o[nm].get("rank_p")}' for nm in sets))
        save()
    res['wall_s'] = time.time() - T0; save(); log(f'wrote {OUT}/{tag}.json; DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tseed', type=int, default=4747); ap.add_argument('--fseed', type=int, default=4748)
    ap.add_argument('--per', type=int, default=3000); ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--nrand-holds', type=int, default=500); ap.add_argument('--nrand-steer', type=int, default=200)
    ap.add_argument('--suffix', default='')
    main(ap.parse_args())
