"""E37y verify: rerun the key arms of E37y on fresh samples and add the controls the node lacks.

Fresh samples: T = W+Z jets of interp.jedi_core.sample(PER, 6161); F = W+Z jets of sample(PER, 6162) minus rows in T (E37y used
5757 / 5758).  v (1-dim DAS direction at h4) is refitted on F with the E37 code, exactly as E37y does.
Arm 1 (replication): steering h4 + c sd_u v, c in {+-0.5, +-1, +-2}; per-jet through-zero slope of the W-Z margin change; matched
  contrast Delta over 80-100 GeV in 5-GeV and 1-GeV bins; permutation p within 1-GeV bins; 100 random h4 directions.
  Added controls:
  (a) small-step slope: c = +-0.5 only (no unit at the clip), to check that Delta is not a product of clipping at c = +-2;
  (b) state matching: W and Z jets compared inside cells of (1-GeV mass bin x baseline W-Z margin bin of 1 log-odds), of
      (1-GeV mass x on/off pattern of z0, z7, z11, z14) and of (1-GeV mass x baseline u bin of 0.5 sd); permutation p within cells.
      Competing hypotheses: H_class (the slope depends on the true class beyond the network's own state) predicts Delta != 0
      inside these cells; H_state (the slope is a function of where the jet sits in the tail -- its baseline margin / which z
      units are on -- and W and Z jets of equal mass differ only in that state) predicts Delta -> 0;
  (c) regression: slope ~ isW + 1-GeV mass dummies + m0 + m0^2 + u0 + u0^2 on 80-100 GeV jets (isW coefficient, bootstrap CI).
Arm 2 (replication): z0, z7, z14 held at unsteered values singly / pairs / jointly; removed share; 200 random d-dim z holds; all
  holds of d other units; Holm over 7.
Arm 3 (replication on a random subset of N3 T jets): E37y width_variants (Ws, Eshuf, Ewid, EmW, Erot at alpha = +-0.5); du, dW_p,
  dlead; R_w with the OLS beta_w (E37y regressors) and with the kNN (log m, log pT) beta_w; per-jet regression of du on dW_p
  inside each alpha (does more widening give lower u at fixed alpha?).
Arm 4 (replication): DAS k = 1, 3 in the complement of span(v, M2) on F Z jets under D(+0.3); test on T Z jets; 200 random
  complement subspaces; hold v + M2 + Q_k vs random (3 + k)-dim holds (100).
Run: ./job.sh e37y-verify 'python -m interp.e37y_verify'   (--smoke K: K jets per class)
Writes $AI_RESULTS/jedi/e37y_verify[_smoke].json.
"""
import argparse
import json
import time
from itertools import combinations

import numpy as np

from interp import e37y_sharing_gate_jedi as Y

E, X, H50 = Y.E, Y.X, Y.H50
OUT = Y.OUT
TSEED, FSEED = 6161, 6162
T0 = time.time()


def log(*a): print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def cell_delta(sl, y, cells, sel, W=None, nmin=3):
    """W-Z contrast of sl inside cells (int array), weights nW nZ / (nW + nZ); restricted to sel"""
    num = den = 0.0; bnum = 0.0; nused = 0; ncells = 0
    for c in np.unique(cells[sel]):
        s = sel & (cells == c); sW = np.flatnonzero(s & (y == 2)); sZ = np.flatnonzero(s & (y == 3))
        if len(sW) < nmin or len(sZ) < nmin: continue
        w = len(sW) * len(sZ) / (len(sW) + len(sZ)); num += w * (sl[sW].mean() - sl[sZ].mean()); den += w; nused += len(sW) + len(sZ); ncells += 1
        if W is not None: bnum = bnum + w * (X.bmean(W, sl, sW) - X.bmean(W, sl, sZ))
    if den == 0: return dict(delta=None, n_jets=0, n_cells=0)
    o = dict(delta=float(num / den), n_jets=int(nused), n_cells=int(ncells))
    if W is not None: bd = bnum / den; o.update(ci=E.qci(bd), p_boot=X.pboot(bd))
    return o


def cell_perm(sl, y, cells, sel, nperm, seed, nmin=3):
    obs = cell_delta(sl, y, cells, sel, nmin=nmin)['delta']; rng = np.random.default_rng(seed); cnt = 0
    if obs is None: return None
    idx = np.flatnonzero(sel); groups = [idx[cells[idx] == c] for c in np.unique(cells[idx])]
    for _ in range(nperm):
        yp = y.copy()
        for g in groups: yp[g] = y[rng.permutation(g)]
        dp_ = cell_delta(sl, yp, cells, sel, nmin=nmin)['delta']; cnt += dp_ is not None and abs(dp_) >= abs(obs)
    return float((1 + cnt) / (1 + nperm))


def main(a):
    from interp.jedi_core import sample
    from interp.x10_das_jedi import JNet, proj
    import jax
    sm = a.smoke; per = sm or a.per; tag = 'e37y_verify' + ('_smoke' if sm else '')
    steps = 60 if sm else 400; nrd = 20 if sm else 100; B = 200 if sm else 2000; nperm = 200 if sm else 1000
    nrh = 20 if sm else 200; nrand_das = 30 if sm else 200; n3 = 150 if sm else a.n3
    res = dict(node='E37y-verify', smoke=sm, per_class=per, seeds=dict(T=TSEED, F=FSEED), B=B, n_perm=nperm, n3=n3)
    save = lambda: (OUT / f'{tag}.json').write_text(json.dumps(E.jsonable(res), indent=1))
    xT, yT, _ = sample(per, TSEED); xF, yF, _ = sample(per, FSEED); rT = E.rows_of(per, TSEED); rF = E.rows_of(per, FSEED)
    xT = np.asarray(xT, np.float32); xF = np.asarray(xF, np.float32)
    net = JNet(); net.off = np.zeros(5); net.shp = 'a'
    res['acc_T_5class'] = float((net.net.run(xT)['logits'].argmax(1) == yT).mean())
    log(f'T {len(yT)} jets, F {len(yF)}; F rows in T {int(np.isin(rF, rT).sum())}; acc T {res["acc_T_5class"]:.4f}')
    DF = E.prep(xF, yF, rF, FSEED, exclude_rows=rT, sm=sm)
    wz = (yT == 2) | (yT == 3); x = xT[wz]; y = yT[wz]; J = len(y); del xT, xF
    P4, nn, LAB, c0, PE, accD, dP = E.make_edits(x, TSEED, dcs=(0.3,), kinds=('D',))
    XT = {'id': E.model_xn(P4, nn, Y.axis(P4, nn))[0]}; PD = PE[(0.3, 'D')]; XT['D'] = E.model_xn(PD, nn, Y.axis(PD, nn))[0]; del PE, PD
    mT = Y.mass_of(P4, nn); res['N'] = dict(T_WZ=J, F_WZ=DF['J'])
    res['mass_counts_T'] = {f'{lo}-{hi}': dict(W=int(((y == 2) & (mT >= lo) & (mT < hi)).sum()), Z=int(((y == 3) & (mT >= lo) & (mT < hi)).sum())) for lo, hi in Y.MB}
    RF = E.reps_all(net, DF); RT = {k: net.reps(v, keep=('h4', 'logits')) for k, v in XT.items()}; del XT
    h0 = RT['id']['h4']; m0 = E.margin(RT['id']['logits']); live = [int(u) for u in np.flatnonzero(h0.std(0) > 1e-9)]; lm = np.isin(np.arange(64), live)
    tail, cWZ = Y.make_tail(net); lo_t, z0 = tail(h0); res['tail_check'] = float(np.abs(lo_t - RT['id']['logits']).max())
    log(f'mass counts {res["mass_counts_T"]}; tail check {res["tail_check"]}')
    # ---- v
    PF = E.build_pairs(DF, RF, 'h4'); fit = [PF['self'], PF['cross']]
    hb = np.concatenate([p.hb for p in fit]); hs = np.concatenate([p.hs for p in fit]); tg = np.concatenate([p.mt for p in fit]); grp = np.concatenate([p.jet for p in fit])
    Q1, info = E.das_fit_margin(net, 'h4', hb, hs, tg, grp, 1, live, seed=TSEED, steps=steps); jax.clear_caches()
    res['F_interchange_self'] = E.metrics(E.patch(net, 'h4', PF['self'].hb, PF['self'].hs, Q1), PF['self'])
    del hb, hs, tg, grp, PF, fit
    v = Q1[:, 0].copy()
    QS = dict(np.load(OUT / 'e37_das_Q.npz')); MS = dict(np.load(OUT / 'e37_das_M.npz'))
    M2 = MS['M2_h4'].copy(); M2[~lm] = 0; M2 = np.linalg.qr(M2)[0]
    v37 = QS['h4_k1'][:, 0].copy(); v37[~lm] = 0; v37 /= np.linalg.norm(v37)
    OF = np.full((DF['J'], len(Y.LQ)), np.nan)
    for j in range(DF['J']):
        if DF['nn'][j] >= 3: OF[j] = Y.lite(DF['P4'][j, :DF['nn'][j]], DF['LAB'][j, :DF['nn'][j]])
    okF = np.isfinite(OF).all(1) & (OF[:, 4] > 1)
    if np.corrcoef(RF['id']['h4'][okF] @ v, OF[okF, 0])[0, 1] < 0: v = -v
    u0 = h0 @ v; sdu = float(u0.std())
    res['direction'] = dict(das=info, cos_vs_E37_stored=float(abs(v @ v37)), max_cos_vs_M2=float(np.linalg.norm(M2.T @ v)), sd_u=sdu,
                            corr_u_zh_F=float(np.corrcoef(RF['id']['h4'][okF] @ v, OF[okF, 0])[0, 1]))
    log('direction ' + json.dumps(E.jsonable(res['direction'])) + ' F interchange ' + json.dumps(E.jsonable(res['F_interchange_self'])))
    W = X.bweights(J, B, 21); WF = X.bweights(DF['J'], 500, 22); rg = net.rng_['h4']; save()

    # ---- Arm 1
    def steer(vv, sd, Pz=None, keepz=False):
        dms, zs = [], []
        for c in Y.CS:
            hp = proj(rg, h0, h0 + c * sd * vv[None, :]); lo, z = tail(hp, Pz, z0 if Pz is not None else None); dms.append(E.margin(lo) - m0)
            if keepz: zs.append(z)
        dms = np.array(dms); cc = np.array(Y.CS) * sd; sl = (cc[:, None] * dms).sum(0) / (cc ** 2).sum()
        sm_ = [i for i, c in enumerate(Y.CS) if abs(c) == 0.5]; sls = (cc[sm_, None] * dms[sm_]).sum(0) / (cc[sm_] ** 2).sum()
        return sl, dms, sls, zs
    sl, dms, sls, _ = steer(v, sdu)
    # clipping at c = +-0.5
    clip05 = max(float((np.abs(proj(rg, h0, h0 + c * sdu * v[None, :]) - (h0 + c * sdu * v[None, :])) > 1e-9).any(1).mean()) for c in (-0.5, 0.5))
    A1 = dict(clip_frac_c05=clip05)
    o, bd_free = Y.cstat(sl, y, mT, W); A1['matched'] = o; A1['perm_p'] = Y.perm_p(sl, y, mT, nperm, 5)
    A1['fine_1GeV'] = Y.fine_delta(sl, y, mT, W); A1['linear_model'] = Y.lin_class(sl, y, mT, W)
    os_, _ = Y.cstat(sls, y, mT, W); A1['small_c_matched'] = {k: os_.get(k) for k in ('delta', 'ci', 'W_matched', 'Z_matched', 'p_boot')}
    A1['small_c_fine_1GeV'] = Y.fine_delta(sls, y, mT, W)
    A1['per_c_matched'] = {f'{c:+}': {k: Y.cstat(dms[i], y, mT)[0].get(k) for k in ('W_matched', 'Z_matched', 'delta')} for i, c in enumerate(Y.CS)}
    # state matching
    win = (mT >= 80) & (mT < 100) & ((y == 2) | (y == 3)); mb1 = np.floor(mT).astype(int)
    zon = (z0[:, [0, 7, 11, 14]] > 0).astype(int) @ np.array([1, 2, 4, 8])
    mb5 = (np.floor(mT / 5) * 5).astype(int)
    cellsets = dict(mass1=mb1, mass5_x_m0=mb5 * 1000 + np.floor(m0).astype(int) + 500, mass1_x_m0=mb1 * 1000 + np.floor(m0).astype(int) + 500,
                    mass1_x_m0half=mb1 * 1000 + np.floor(2 * m0).astype(int) + 500,
                    mass1_x_zpattern=mb1 * 100 + zon,
                    mass1_x_u0=mb1 * 1000 + np.floor(u0 / (0.5 * sdu)).astype(int) + 500,
                    mass1_x_m0_x_u0=(mb1 * 1000 + np.floor(m0).astype(int) + 500) * 1000 + np.floor(u0 / sdu).astype(int) + 500)
    SM = {}
    for nm, cl in cellsets.items():
        r = cell_delta(sl, y, cl, win, W); r['perm_p'] = cell_perm(sl, y, cl, win, nperm, 6)
        r['small_c'] = cell_delta(sls, y, cl, win, W); SM[nm] = r
        log(f'  state matching {nm}: {json.dumps(E.jsonable(r))}')
    A1['state_matching'] = SM
    # baseline-state differences at matched mass (descriptive)
    A1['baseline_m0_matched'] = cell_delta(m0, y, mb1, win, W); A1['baseline_u0_matched'] = cell_delta(u0, y, mb1, win, W)
    # regression with state controls
    s = np.flatnonzero(win); isw = (y[s] == 2).astype(float); Dm = (mb1[s][:, None] == np.unique(mb1[s])[None, :]).astype(float)
    def reg(cols):
        Xm = np.column_stack([isw, Dm] + list(cols)); b = np.linalg.lstsq(Xm, sl[s], rcond=None)[0][0]; bb = []
        for w in W[:500].astype(np.float64):
            ww = np.sqrt(w[s]); bb.append(np.linalg.lstsq(Xm * ww[:, None], sl[s] * ww, rcond=None)[0][0])
        return dict(isW=float(b), ci=E.qci(bb), p_boot=X.pboot(bb))
    A1['regression'] = dict(mass_only=reg([]), mass_m0=reg([m0[s], m0[s] ** 2]), mass_u0=reg([u0[s], u0[s] ** 2]),
                            mass_m0_u0=reg([m0[s], m0[s] ** 2, u0[s], u0[s] ** 2, m0[s] * u0[s]]),
                            mass_m0_u0_z=reg([m0[s], m0[s] ** 2, u0[s], u0[s] ** 2, m0[s] * u0[s], ] + [np.maximum(z0[s, k], 0) for k in range(16)]))
    log(f'arm 1: matched {o["delta"]} {o.get("ci")} W {o["W_matched"]:.3f} Z {o["Z_matched"]:.3f}; perm p {A1["perm_p"]}; fine {A1["fine_1GeV"]}; '
        f'small c {A1["small_c_matched"]}; small c fine {A1["small_c_fine_1GeV"]}; regression {json.dumps(E.jsonable(A1["regression"]))}; '
        f'baseline m0 diff {A1["baseline_m0_matched"]}')
    rr = np.random.default_rng(77); rd = []
    for i in range(nrd):
        g = E.rand_live(rr, live, 64, 1)[:, 0]; sdg = float((h0 @ g).std()); slg = steer(g, sdg)[0]; rd.append(Y.cstat(slg * sdg, y, mT)[0]['delta'])
    rd = np.array(rd, float); dv = o['delta'] * sdu
    A1['random_dirs'] = dict(n=nrd, v_delta_per_sd=dv, q=np.quantile(rd, [.025, .5, .975]).tolist(), rank_p_abs=float((1 + (np.abs(rd) >= abs(dv)).sum()) / (1 + nrd)))
    log(f'  random dirs {A1["random_dirs"]}')
    res['arm1'] = A1; save()

    # ---- Arm 2
    nullR = {}; rr = np.random.default_rng(78)
    for d in (1, 2, 3):
        nullR[d] = np.array([1 - Y.delta_fast(steer(v, sdu, Pz=Y.rand_proj(rr, list(range(16)), 16, d))[0], y, mT) / o['delta'] for _ in range(nrh)])
    others = [k for k in range(16) if k not in (0, 7, 14)]; nullA = {}
    for d in (1, 2, 3):
        vals = []
        for comb in combinations(others, d):
            Pz = np.zeros((16, 16)); Pz[list(comb), list(comb)] = 1; vals.append(1 - Y.delta_fast(steer(v, sdu, Pz=Pz)[0], y, mT) / o['delta'])
        nullA[d] = np.array(vals)
    named = {}; ps = []
    for nm, units in Y.NAMED.items():
        Pz = np.zeros((16, 16)); Pz[units, units] = 1; slh, _, slsh, _ = steer(v, sdu, Pz=Pz); oh, bd = Y.cstat(slh, y, mT, W); d = len(units)
        sh = 1 - oh['delta'] / o['delta']; bs = 1 - bd / bd_free
        r = dict(share=float(sh), share_ci=E.qci(bs), p_boot_le_half=float(np.mean(bs <= 0.5)), delta_held=oh['delta'],
                 rank_p_rand=float((1 + (nullR[d] >= sh).sum()) / (1 + nrh)), rank_p_units=float((1 + (nullA[d] >= sh).sum()) / (1 + len(nullA[d]))),
                 share_small_c=float(1 - Y.cstat(slsh, y, mT)[0]['delta'] / os_['delta']),
                 share_state_matched_m0=float(1 - (cell_delta(slh, y, cellsets['mass1_x_m0'], win)['delta'] or np.nan) / (SM['mass1_x_m0']['delta'] or np.nan)) if SM['mass1_x_m0']['delta'] else None)
        named[nm] = r; ps.append(r['rank_p_rand'])
    for (nm, r), p_ in zip(named.items(), E.holm(ps)): r['holm'] = float(p_)
    res['arm2'] = dict(named=named, null_rand={d: np.quantile(v_, [.5, .95]).tolist() for d, v_ in nullR.items()},
                       null_units={d: dict(q95=float(np.quantile(v_, .95)), max=float(v_.max()), n=len(v_)) for d, v_ in nullA.items()})
    log('arm 2: ' + json.dumps(E.jsonable(res['arm2'])))
    save()

    # ---- Arm 4 (before arm 3: cheap)
    Bv = np.linalg.qr(np.c_[v, M2])[0]; Pc = np.eye(64) - Bv @ Bv.T
    sF = np.flatnonzero(DF['acc'][0.3] & (DF['y'] == 3)); hbF, hsF = RF['id']['h4'][sF], RF[(0.3, 'D')]['h4'][sF]; tgF = E.margin(RF[(0.3, 'D')]['logits'][sF])
    sT = np.flatnonzero(accD[0.3] & (y == 3)); PTz = E.PairSet(h0[sT], RT['D']['h4'][sT], RT['D']['logits'][sT], RT['id']['logits'][sT], np.full(len(sT), 3), sT)
    mfree = E.margin(RT['D']['logits'][sT]) - m0[sT]
    A4 = dict(n_fit=int(len(sF)), n_test=int(len(sT)), free_dmargin=float(mfree.mean()), copy_v=E.metrics(E.patch(net, 'h4', PTz.hb, PTz.hs, v[:, None]), PTz))

    def hold_share(Q):
        h = proj(rg, PTz.hs, PTz.hs + ((PTz.hb - PTz.hs) @ Q) @ Q.T); mh = E.margin(net.exact('h4', h)[0]) - m0[sT]; return float(1 - mh.mean() / mfree.mean())
    A4['hold_v_M2'] = hold_share(Bv); rr = np.random.default_rng(79); das = {}
    for k in (1, 3):
        Qk, inf = Y.das_fit_proj(net, hbF, hsF, tgF, sF, k, live, Pc, seed=TSEED + k, steps=steps)
        mt = E.metrics(E.patch(net, 'h4', PTz.hb, PTz.hs, Qk), PTz); mt.update(E.metrics(E.patch(net, 'h4', PTz.hb, PTz.hs, Qk), PTz, W)); mt['fit'] = inf
        rf, ri = [], []
        for _ in range(nrand_das):
            G = np.zeros((64, k)); G[live] = rr.standard_normal((len(live), k)); G = np.linalg.qr(Pc @ G)[0]
            mr = E.metrics(E.patch(net, 'h4', PTz.hb, PTz.hs, G), PTz); rf.append(mr['frac']); ri.append(mr['iia_wz'] if mr['iia_wz'] is not None else np.nan)
        rf, ri = np.array(rf, float), np.array(ri, float)
        mt['random'] = dict(frac_q95=float(np.nanquantile(rf, .95)), iia_q95=float(np.nanquantile(ri, .95)))
        mt['rank_p_iia'] = float((1 + np.nansum(ri >= (mt['iia_wz'] or 0))) / (1 + nrand_das))
        sh = hold_share(np.linalg.qr(np.c_[Bv, Qk])[0]); hn = []
        for _ in range(nrh // 2):
            G = np.zeros((64, 3 + k)); G[live] = rr.standard_normal((len(live), 3 + k)); hn.append(hold_share(np.linalg.qr(G)[0]))
        mt['hold_v_M2_Q'] = dict(share=sh, random_q95=float(np.quantile(hn, .95)), rank_p=float((1 + (np.array(hn) >= sh).sum()) / (1 + len(hn))))
        das[f'k{k}'] = mt; log(f'arm 4 k={k}: ' + json.dumps(E.jsonable(mt)))
    A4['das'] = das; res['arm4'] = A4; del RF; save()

    # ---- Arm 3 on a random subset
    t = time.time(); sub = np.sort(np.random.default_rng(31).choice(np.flatnonzero(nn >= 3), min(n3, int((nn >= 3).sum())), replace=False))
    keys = [(k, al) for al in Y.AL for k in ('Ws', 'Eshuf', 'Ewid', 'EmW', 'Erot')]; Js = len(sub)
    P4s, nns = P4[sub], nn[sub]; PV = {k: P4s.copy() for k in keys}; ACC = {k: np.zeros(Js, bool) for k in keys}
    OB = {k: np.full((Js, len(Y.LQ)), np.nan) for k in keys}; OT = np.full((Js, len(Y.LQ)), np.nan)
    for i, j in enumerate(sub):
        n = int(nn[j]); P = P4[j, :n].astype(np.float64); lab = LAB[j, :n]; OT[i] = Y.lite(P, lab)
        outv, inf, _, _ = Y.width_variants(P, lab, TSEED, j)
        for k, Q in outv.items(): PV[k][i, :n] = Q; ACC[k][i] = True; OB[k][i] = Y.lite(Q, lab)
    log(f'arm 3 variants built for {Js} jets [{time.time() - t:.0f}s]')
    u0s = u0[sub]; DU = {}
    for k in keys:
        xv = E.model_xn(PV[k], nns, Y.axis(PV[k], nns))[0]; DU[k] = net.reps(xv, keep=('h4',))['h4'] @ v - u0s
    Fi = np.flatnonzero(okF)
    res['arm3_note'] = 'beta_w fitted on F jets (E37y regressors)'
    A3 = dict(n=Js, accepted={f'{k[0]}{k[1]:+}': float(ACC[k].mean()) for k in keys})
    res['arm3'] = A3
    # natural slope beta_w on F (needs F h4 of unedited jets: recomputed)
    hF = net.reps(DF['X']['id'], keep=('h4',))['h4']; uF = hF[Fi] @ v; O = OF[Fi]; lmF = np.log(O[:, 4])
    XF = np.stack([np.ones(len(Fi)), lmF, lmF ** 2, np.log(O[:, 5]), O[:, 0], O[:, 0] ** 2, O[:, 1], O[:, 2], O[:, 3], O[:, 6]], 1)
    bw = float(Y.wols(XF, uF)[-1]); bwb = np.array([Y.wols(XF, uF, w[Fi].astype(np.float64))[-1] for w in WF])
    (ru, rw), _, _ = X.knn_resid(np.c_[lmF, np.log(O[:, 5])], [uF, O[:, 6]]); bk = float((ru * rw).sum() / (rw ** 2).sum())
    A3['beta_w'] = dict(ols=bw, ci=E.qci(bwb), knn=bk)
    Ws_ = X.bweights(Js, 1000, 23); R = {}
    for al in Y.AL:
        for ed, ct in (('Ws', 'Erot'), ('Ws', 'Eshuf'), ('Ewid', 'EmW')):
            s = np.flatnonzero(ACC[(ed, al)] & ACC[(ct, al)])
            if len(s) < 10: continue
            ddu = DU[(ed, al)] - DU[(ct, al)]; dw = OB[(ed, al)][:, 6] - OB[(ct, al)][:, 6]; dl = OB[(ed, al)][:, 7] - OB[(ct, al)][:, 7]
            bu = X.bmean(Ws_, ddu, s)
            r = dict(n=int(len(s)), du=float(ddu[s].mean()), du_ci=E.qci(bu), dWp=float(dw[s].mean()), dWp_rel=float((dw[s] / OT[s, 6]).mean()), dlead=float(dl[s].mean()),
                     R_w_ols=float(ddu[s].mean() / (bw * dw[s].mean())), R_w_knn=float(ddu[s].mean() / (bk * dw[s].mean())))
            if ed == 'Ws':  # within-alpha per-jet relation: more widening -> lower u?
                Xr = np.c_[np.ones(len(s)), dw[s] / O[:, 6].std()]; b = np.linalg.lstsq(Xr, ddu[s], rcond=None)[0]
                r['within_alpha_du_per_sd_dWp'] = float(b[1]); r['within_alpha_corr'] = float(np.corrcoef(dw[s], ddu[s])[0, 1])
                r['frac_jets_dWp_pos'] = float((dw[s] > 0).mean())
            R[f'{ed}_vs_{ct}|{al:+}'] = r
    A3['R'] = R
    # sign test of the width reading: both alphas widen at fixed prong mass; a width effect predicts the same sign of du
    log('arm 3: ' + json.dumps(E.jsonable(A3)))
    res['wall_s'] = time.time() - T0; save(); log(f'wrote {OUT / tag}.json; DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', type=int, default=0); ap.add_argument('--per', type=int, default=2500)
    ap.add_argument('--n3', type=int, default=1200)
    main(ap.parse_args())
