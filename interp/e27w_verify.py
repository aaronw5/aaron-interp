"""E27w verification (fresh samples + missing controls).  Reuses the E27w edit code (interp.e27w_mass_vs_m2_jedi).

--arm AB : fit = sample(per_class=1000, seed=2741) (fresh; E27w used 2701 / 2702).
  Arm A as E27w (rigid +0.06 rad detarel vs M2-matched dilation).
  Arm B hardest 14, m -2 / -5 / -10 %: B (turn toward M1), ctrl (random-sign turn, M1 restored, D matched), dil_m, B_frz, dil_frz,
  and the new drive-matched control ctrl_drv: the same random-sign + M1-restoring turn, with the displacement target D scaled by
  k (k from 1.0..3.0, the k whose mean |dP| over the nine half-plane objects is closest to B's), so the control moves the angular
  (half-plane) first-round objects as much as B while keeping m fixed.  Tests: sign-flip on per-jet h4 read-out differences
  (B vs ctrl, B vs ctrl_drv, B vs dil_m), McNemar on Z->W and W->Z (B vs ctrl, B vs ctrl_drv, B vs dil_m).
--arm C : fit = sample(1000, 2741), test = sample(1000, 2742) minus overlap with the fit sample.  Arm C as E27w (60 / 20 random
  smooth radial reshapings, H2 and Hm), plus
  - single power with the coefficient fixed to 2/beta (a network that reads exactly sum pT r^beta must respond to the 5 % M2
    dilation, X_beta = beta/2, with R_marg = 1),
  - a single-regressor null: 66 random smooth radial sums sum pT h(r) per draw (as many candidates as the beta grid), the one
    with the lowest fit MSE is kept and scored on the test edits; 200 draws.
Outputs: $AI_RESULTS/jedi/e27w_verify_AB.json, e27w_verify_C.json.
  python -m interp.e27w_verify --arm AB|C [--smoke 60]
"""
import argparse
import json
import time

import numpy as np

from interp import e27w_mass_vs_m2_jedi as E

T0 = time.time(); W_, Z_ = 2, 3
SEED_FIT, SEED_TEST = 2741, 2742
KGRID = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)


def log(*a): print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def arm_AB(a, net, sample, clean, holm, mass, RidgeR, mcnemar):
    pc = a.smoke or 1000; NB = 200 if a.smoke else 1000; NP = 1000 if a.smoke else 10000
    JS = E.OUT / ('e27w_verify_AB' + ('_smoke' if a.smoke else '') + '.json')
    R = dict(sample=f'sample(per_class={pc}, seed={SEED_FIT})'); save = lambda: JS.write_text(json.dumps(clean(R), indent=1))
    fwd = lambda x, ov=None, keep=('logits', 'h4', 'P', 'g', 'S'): net.run_jetwise(x, ov, keep=keep, batch=1000) if ov else net.run(x, keep=keep, batch=1000)
    x, y, _ = sample(pc, SEED_FIT); J = len(y)
    x64 = np.asarray(x, np.float64); pt = x64[..., 0]; real = pt > 0; r = np.hypot(x64[..., 1], x64[..., 2]); th = np.arctan2(x64[..., 2], x64[..., 1])
    B0 = fwd(x); c0 = B0['logits'].argmax(1); marg0 = B0['logits'][:, Z_] - B0['logits'][:, W_]
    R['N'] = J; R['acc'] = float((c0 == y).mean()); log(f'N {J} accuracy {R["acc"]:.4f}')
    logm0 = np.log(np.maximum(mass(x), 1e-3)); pr = np.random.default_rng(2744).permutation(J); hA, hB = np.sort(pr[:J // 2]), np.sort(pr[J // 2:])
    nf = 2 * len(hA) // 3; RO = RidgeR(B0['h4'], logm0, hA[:nf], hA[nf:]); ro0 = RO(B0['h4'])
    R['readout_r2_heldout_h4'] = float(1 - ((RO(B0['h4'][hB]) - logm0[hB]) ** 2).mean() / logm0[hB].var())
    isB = np.zeros(J, bool); isB[hB] = True
    sdP = B0['P'].std(0) + 1e-12; sdS = B0['S'].std(0) + 1e-12
    Wb = np.random.default_rng(2745).poisson(1.0, (NB, J)).astype(np.float64)
    bm = lambda w, v: (w @ v) / np.maximum(w.sum(1), 1e-12)

    def resp(P):
        ce = P['logits'].argmax(1)
        o = dict(call=ce, ch=ce != c0, wz=(c0 == W_) & (ce == Z_), zw=(c0 == Z_) & (ce == W_), dmarg=P['logits'][:, Z_] - P['logits'][:, W_] - marg0,
                 dh4=RO(P['h4']) - ro0)
        if 'P' in P:
            o['hp_j'] = np.mean(np.abs(P['P'][:, E.HP] - B0['P'][:, E.HP]) / sdP[E.HP], 1)
            o['s31_j'] = np.abs(P['S'][:, 31] - B0['S'][:, 31]) / sdS[31]
            o['srad_j'] = np.mean(np.abs(P['S'][:, E.RADS] - B0['S'][:, E.RADS]) / sdS[E.RADS], 1)
        return o

    def ratio(v, vr, m):
        if m.sum() == 0: return None, [None, None]
        w = Wb[:, m]; return float(v[m].mean() / vr[m].mean()), E.qci(bm(w, v[m]) / bm(w, vr[m]))

    def share(v, m):
        if m.sum() == 0: return None, [None, None]
        return float(v[m].mean()), E.qci(bm(Wb[:, m], v[m].astype(float)))

    def block(Ed, ref, ks):
        o = dict(n=int(ks.sum()), n_W=int((ks & (c0 == W_)).sum()), n_Z=int((ks & (c0 == Z_)).sum()))
        o['R_h4'], o['R_h4_ci'] = ratio(Ed['dh4'], ref['dh4'], ks & isB)
        o['R_marg_Z'], o['R_marg_Z_ci'] = ratio(Ed['dmarg'], ref['dmarg'], ks & (c0 == Z_))
        o['R_marg_W'], o['R_marg_W_ci'] = ratio(Ed['dmarg'], ref['dmarg'], ks & (c0 == W_))
        o['calls_changed'], o['calls_changed_ci'] = share(Ed['ch'], ks)
        o['WZ'], o['WZ_ci'] = share(Ed['wz'], ks & (c0 == W_)); o['ZW'], o['ZW_ci'] = share(Ed['zw'], ks & (c0 == Z_))
        if 'hp_j' in Ed:
            o['drive'] = dict(HP_dP=float(Ed['hp_j'][ks].mean()), S31=float(Ed['s31_j'][ks].mean()), S_radial=float(Ed['srad_j'][ks].mean()))
        return o

    # arm A
    xa = np.array(x, np.float32); xa[..., 1] += np.float32(0.06) * real
    M2 = (pt * r ** 2).sum(1); M2a = (pt * ((x64[..., 1] + 0.06) ** 2 + x64[..., 2] ** 2) * real).sum(1)
    xd = E.to_x(x, real, r * np.sqrt(M2a / M2)[:, None], th); EA, ED = resp(fwd(xa)), resp(fwd(xd)); allk = np.ones(J, bool)
    R['armA'] = dict(translation=block(EA, ED, allk), dil_M2=block(ED, ED, allk), mcnemar_WZ_vs_dil=mcnemar(EA['wz'][c0 == W_], ED['wz'][c0 == W_]),
                     median_dlogm_true=float(np.median(np.log(mass(xa) / mass(x)))))
    log(f'arm A: W->Z {R["armA"]["translation"]["WZ"]:.3f} vs dil {R["armA"]["dil_M2"]["WZ"]:.3f}; R_h4 {R["armA"]["translation"]["R_h4"]:+.3f}'); save()

    # arm B, hardest 14
    rank = np.argsort(np.argsort(-np.where(real, pt, -1), 1), 1); sub = real & (rank < 14)
    M10 = E.M1v(pt, r, th); phu = np.arctan2(M10[:, 1], M10[:, 0]); mx0 = mass(x)
    frz = {'g': (np.ones(64), B0['g']), 'S': (np.isin(np.arange(64), E.CTX).astype(float), B0['S'])}
    R['armB'] = {}; P = {k: [] for k in ('ro_B_ctrl', 'ro_B_drv', 'ro_B_dil', 'ZW_B_ctrl', 'ZW_B_drv', 'ZW_B_dil', 'WZ_B_ctrl', 'WZ_B_drv')}; vn = []
    for f in E.DOSES:
        t = time.time(); v = f'hard14_{int(round(f * 100))}'; vn.append(v)
        d = np.where(sub, E.wrap(phu[:, None] - th), 0.0); tB, okB = E.solve_B(pt, r, th, d, f); thB = th + tB[:, None] * d; DB = E.disp(r, th, thB)
        rng_seed = 27400 + int(f * 100)
        thC, okC = E.solve_ctrl(pt, r, th, sub, tB[:, None] * d, DB, np.random.default_rng(rng_seed))
        xB, xC = E.to_x(x, real, r, thB), E.to_x(x, real, r, thC)
        mB = mass(xB); aD = mB / np.maximum(mx0, 1e-9)
        for _ in range(5):
            xDm = E.to_x(x, real, r * aD[:, None], th); aD = aD * mB / np.maximum(mass(xDm), 1e-9)
        xDm = E.to_x(x, real, r * aD[:, None], th)
        EB, EC_, ED_ = resp(fwd(xB)), resp(fwd(xC)), resp(fwd(xDm))
        base_ok = okB & okC & np.isfinite(np.log(mB / mx0)) & (np.abs(np.log(mass(xC) / mx0)) <= 0.1 * abs(np.log(1 - f)))
        # drive-matched control: scale D by k, pick k whose half-plane drive is closest to B's on base-kept jets
        scan = {}
        for k in KGRID:
            thK, okK = E.solve_ctrl(pt, r, th, sub, k * tB[:, None] * d, k * DB, np.random.default_rng(rng_seed))
            xK = E.to_x(x, real, r, thK); PK = fwd(xK, keep=('P',)); hp = np.mean(np.abs(PK['P'][:, E.HP] - B0['P'][:, E.HP]) / sdP[E.HP], 1)
            mk = base_ok & okK & (np.abs(np.log(mass(xK) / mx0)) <= 0.1 * abs(np.log(1 - f)))
            scan[k] = dict(hp=float(hp[mk].mean()) if mk.any() else np.inf, n_ok=int(mk.sum()), th=thK, ok=okK)
        hpB = float(EB['hp_j'][base_ok].mean()); kbest = min(KGRID, key=lambda k: abs(scan[k]['hp'] - hpB))
        thK, okK = scan[kbest]['th'], scan[kbest]['ok']; xK = E.to_x(x, real, r, thK)
        ks = base_ok & okK & (np.abs(np.log(mass(xK) / mx0)) <= 0.1 * abs(np.log(1 - f)))
        EK = resp(fwd(xK))
        Ed = dict(B=EB, ctrl=EC_, ctrl_drv=EK, dil_m=ED_, B_frz=resp(fwd(xB, ov=frz)), dil_frz=resp(fwd(xDm, ov=frz)), ctrl_drv_frz=resp(fwd(xK, ov=frz)))
        o = dict(dose=f, kept=int(ks.sum()), kept_by_class=np.bincount(y[ks], minlength=5).tolist(), k_drive=kbest,
                 drive_scan={str(k): dict(hp=scan[k]['hp'], n_ok=scan[k]['n_ok']) for k in KGRID}, hp_B_base=hpB,
                 checks=dict(max_abs_dr=dict(B=float(np.max(np.abs(np.hypot(xB[..., 1].astype(float), xB[..., 2]) - r)[ks])),
                                             ctrl_drv=float(np.max(np.abs(np.hypot(xK[..., 1].astype(float), xK[..., 2]) - r)[ks]))),
                             median_dlogm=dict(B=float(np.median(np.log(mB / mx0)[ks])), ctrl=float(np.median(np.log(mass(xC) / mx0)[ks])),
                                               ctrl_drv=float(np.median(np.log(mass(xK) / mx0)[ks])), dil_m=float(np.median(np.log(mass(xDm) / mx0)[ks]))),
                             median_D=dict(B=float(np.median(DB[ks])), ctrl_drv=float(np.median(E.disp(r, th, thK)[ks]))),
                             max_dlogM2=float(np.max(np.abs(np.log((pt * (xB[..., 1].astype(float) ** 2 + xB[..., 2].astype(float) ** 2)).sum(1) / M2))[ks]))))
        for k_ in Ed: o[k_] = block(Ed[k_], ED_, ks)
        mB_ = ks & isB; dR = ED_['dh4'][mB_]; w = Wb[:, mB_]
        for nm, ref in (('ctrl', EC_), ('ctrl_drv', EK)):
            o[f'diff_R_h4_B_minus_{nm}'] = float((EB['dh4'][mB_] - ref['dh4'][mB_]).mean() / dR.mean())
            o[f'diff_R_h4_B_minus_{nm}_ci'] = E.qci((bm(w, EB['dh4'][mB_]) - bm(w, ref['dh4'][mB_])) / bm(w, dR))
        kz, kw = ks & (c0 == Z_), ks & (c0 == W_)
        T = dict(ro_B_ctrl=E.signflip_p(EB['dh4'][mB_] - EC_['dh4'][mB_], NP, 1), ro_B_drv=E.signflip_p(EB['dh4'][mB_] - EK['dh4'][mB_], NP, 2),
                 ro_B_dil=E.signflip_p(EB['dh4'][mB_] - dR, NP, 3),
                 ZW_B_ctrl=mcnemar(EB['zw'][kz], EC_['zw'][kz]), ZW_B_drv=mcnemar(EB['zw'][kz], EK['zw'][kz]), ZW_B_dil=mcnemar(EB['zw'][kz], ED_['zw'][kz]),
                 WZ_B_ctrl=mcnemar(EB['wz'][kw], EC_['wz'][kw]), WZ_B_drv=mcnemar(EB['wz'][kw], EK['wz'][kw]))
        o['tests'] = T
        for k_ in P: P[k_].append(T[k_]['p'] if isinstance(T[k_], dict) else T[k_])
        R['armB'][v] = o
        log(f'{v} ({time.time() - t:.0f} s): kept {ks.sum()}, k_drive {kbest} (HP B {o["B"]["drive"]["HP_dP"]:.3f}, ctrl {o["ctrl"]["drive"]["HP_dP"]:.3f}, '
            f'drv {o["ctrl_drv"]["drive"]["HP_dP"]:.3f}); R_h4 B {o["B"]["R_h4"]:+.3f} ctrl {o["ctrl"]["R_h4"]:+.3f} drv {o["ctrl_drv"]["R_h4"]:+.3f} Bfrz {o["B_frz"]["R_h4"]:+.3f}; '
            f'Z->W B {o["B"]["ZW"]:.3f} ctrl {o["ctrl"]["ZW"]:.3f} drv {o["ctrl_drv"]["ZW"]:.3f} dil {o["dil_m"]["ZW"]:.3f}; W->Z B {o["B"]["WZ"]:.3f} ctrl {o["ctrl"]["WZ"]:.3f} drv {o["ctrl_drv"]["WZ"]:.3f}')
        save()
    R['holm'] = {k_: dict(zip(vn, holm(v_).tolist())) for k_, v_ in P.items()}; R['seconds'] = time.time() - T0; save(); log('done', JS)


def arm_C(a, net, sample, clean, holm, mass):
    sm = a.smoke; pc = sm or 1000; NB = 100 if sm else 300
    NDF, NDT, NNULL = (8, 4, 20) if sm else (60, 20, 200)
    JS = E.OUT / ('e27w_verify_C' + ('_smoke' if sm else '') + '.json')
    R = dict(samples=dict(fit=f'sample({pc}, {SEED_FIT})', test=f'sample({pc}, {SEED_TEST}) minus overlap')); save = lambda: JS.write_text(json.dumps(clean(R), indent=1))
    x, y, _ = sample(pc, SEED_FIT); xt, yt, _ = sample(pc, SEED_TEST)
    hf = {hash(v.tobytes()) for v in x}; kt = np.array([hash(v.tobytes()) not in hf for v in xt]); xt, yt = xt[kt], yt[kt]
    R['N_fit'], R['N_test'] = len(y), len(yt)
    S = {}
    for sn, xx in (('fit', x), ('test', xt)):
        x64 = np.asarray(xx, np.float64); lo = net.run(xx, keep=('logits',), batch=1000)['logits']
        S[sn] = dict(x=xx, pt=x64[..., 0], r=np.hypot(x64[..., 1], x64[..., 2]), th=np.arctan2(x64[..., 2], x64[..., 1]), c0=lo.argmax(1), marg0=lo[:, Z_] - lo[:, W_])
    R['acc_fit'] = float((S['fit']['c0'] == y).mean()); R['acc_test'] = float((S['test']['c0'] == yt).mean()); log(f'acc fit {R["acc_fit"]:.4f} test {R["acc_test"]:.4f}')
    sdb = E.basis(S['fit']['r'])[S['fit']['pt'] > 0].std(0)
    rngd = {'fit': np.random.default_rng(27413), 'test': np.random.default_rng(27423)}
    BUMPV = np.random.default_rng(2748).standard_normal((NNULL, 66, 12))     # single-regressor null: 66 candidates per draw
    VN = np.random.default_rng(2749).standard_normal((NNULL, 12, 5))
    AC = {}
    for sn, nd in (('fit', NDF), ('test', NDT)):
        D_ = S[sn]; P_, r_, th_ = D_['pt'], D_['r'], D_['th']; re_ = P_ > 0; pop = D_['c0'] == W_
        Pp, rp, rep = P_[pop], r_[pop], re_[pop]
        lam0 = E.moments(Pp, rp, rep, E.BGRID); m0_ = mass(D_['x'][pop]); phi0 = E.bumpfeat(Pp, rp, rep)
        ld = net.run(E.to_x(D_['x'], re_, r_ * np.sqrt(1.05), th_), keep=('logits',), batch=1000)['logits']; dmd = (ld[:, Z_] - ld[:, W_] - D_['marg0'])[pop]
        rows = dict(dmarg=[], keep=[], dlam=[], dm=[], dnul=[], set=[])
        for k in range(nd):
            w = rngd[sn].standard_normal(5); w /= np.linalg.norm(w); f = E.make_f(w, r_, re_, sdb)
            for hold in ('H2', 'Hm'):
                r2_, ok = E.edit_C(P_, r_, th_, re_, f, hold); xe = E.to_x(D_['x'], re_, r2_, th_)
                lo = net.run(xe, keep=('logits',), batch=1000)['logits']; rep2 = r2_[pop]; kp = ok[pop]
                rows['dmarg'].append((lo[:, Z_] - lo[:, W_] - D_['marg0'])[pop]); rows['keep'].append(kp)
                rows['dlam'].append(np.log(E.moments(Pp, rep2, rep, E.BGRID) / lam0)); rows['dm'].append(np.log(np.maximum(mass(xe[pop]), 1e-9) / m0_))
                rows['dnul'].append((E.bumpfeat(Pp, rep2, rep) - phi0)[kp].mean(0) if kp.any() else np.zeros(12)); rows['set'].append(hold)
            if k % 20 == 0: log(f'arm C {sn} {k}/{nd}')
        AC[sn] = dict(dmd=dmd, **{k_: (np.nan_to_num(np.array(v)) if k_ != 'set' else v) for k_, v in rows.items()})
    ib = {b: int(np.argmin(np.abs(E.BGRID - b))) for b in E.BETAS}

    def agg(sn, wts):
        A = AC[sn]; K = A['keep'] * wts[None]; den = np.maximum(K.sum(1), 1e-12)
        yv = (K * A['dmarg']).sum(1) / (K * A['dmd'][None]).sum(1)
        G = np.einsum('kj,kjb->kb', K, A['dlam']) / den[:, None] / E.L105; Mm = 2 * (K * A['dm']).sum(1) / den / E.L105
        return yv, G, Mm
    nf_, nt_ = len(AC['fit']['dmd']), len(AC['test']['dmd'])
    yf, Gf, Mf = agg('fit', np.ones(nf_)); yt_, Gt, Mt = agg('test', np.ones(nt_))
    MODELS = dict(M2=lambda G, M: G[:, [ib[2.0]]], m=lambda G, M: M[:, None], M2_beta1=lambda G, M: G[:, [ib[2.0], ib[1.0]]],
                  family=lambda G, M: G[:, [ib[b] for b in E.BETAS]])
    C = dict(n_edits_fit=len(yf), n_edits_test=len(yt_), n_W_fit=nf_, n_W_test=nt_, R_marg_range_fit=[float(yf.min()), float(yf.max())],
             R_marg_range_H2_fit=[float(yf[::2].min()), float(yf[::2].max())], X_beta2_range_fit=[float(Gf[:, ib[2.0]].min()), float(Gf[:, ib[2.0]].max())],
             X_m_range_fit=[float(Mf.min()), float(Mf.max())], models={})
    alph = {}
    for nm, fn in MODELS.items():
        Xf, Xt = fn(Gf, Mf), fn(Gt, Mt); al, _ = E.cv_alpha(Xf, yf); co = E.ridge(Xf, yf, al); alph[nm] = al
        C['models'][nm] = dict(alpha=al, coef=co.tolist(), r2_fit=E.r2(yf, Xf @ co), r2_test=E.r2(yt_, Xt @ co))

    def single(yv, G, fixed):
        """best beta on fit edits: free coefficient (ridge ~0) or coefficient fixed to 2/beta; returns (gi, coef, mse list)"""
        mse = []
        for gi in range(len(E.BGRID)):
            X = G[:, gi]; c = 2 / E.BGRID[gi] if fixed else float(X @ yv / max(X @ X, 1e-30)); mse.append(((yv - c * X) ** 2).mean())
        gi = int(np.argmin(mse)); X = G[:, gi]; c = 2 / E.BGRID[gi] if fixed else float(X @ yv / max(X @ X, 1e-30)); return gi, c, mse
    for nm, fixed in (('single_power', False), ('single_power_coef_2_over_beta', True)):
        gi, c, mse = single(yf, Gf, fixed)
        C['models'][nm] = dict(beta=float(E.BGRID[gi]), coef=c, implied_dilation_R=c * E.BGRID[gi] / 2, r2_fit=E.r2(yf, c * Gf[:, gi]), r2_test=E.r2(yt_, c * Gt[:, gi]),
                               mse_by_beta={f'{b:g}': float(v) for b, v in zip(E.BGRID, mse)})
    # nulls
    Nf5 = np.einsum('kb,qbv->kqv', AC['fit']['dnul'], VN); Nt5 = np.einsum('kb,qbv->kqv', AC['test']['dnul'], VN)
    n5 = []
    for q in range(NNULL):
        Xf, Xt = Nf5[:, q], Nt5[:, q]; sc = np.abs(Xf).std(0) + 1e-30; al, _ = E.cv_alpha(Xf / sc, yf); co = E.ridge(Xf / sc, yf, al); n5.append(E.r2(yt_, (Xt / sc) @ co))
    n1 = []; Nf1 = np.einsum('kb,qcb->kqc', AC['fit']['dnul'], BUMPV); Nt1 = np.einsum('kb,qcb->kqc', AC['test']['dnul'], BUMPV)
    for q in range(NNULL):
        best = None
        for c_ in range(BUMPV.shape[1]):
            X = Nf1[:, q, c_]; co = X @ yf / max(X @ X, 1e-30); e = ((yf - co * X) ** 2).mean()
            if best is None or e < best[0]: best = (e, c_, co)
        n1.append(E.r2(yt_, best[2] * Nt1[:, q, best[1]]))
    n5, n1 = np.array(n5), np.array(n1)
    C['null_5_random_smooth'] = dict(median=float(np.median(n5)), q95=float(np.quantile(n5, .95)), p_family=float((1 + (n5 >= C['models']['family']['r2_test']).sum()) / (1 + len(n5))))
    C['null_1_random_smooth_best_of_66'] = dict(median=float(np.median(n1)), q95=float(np.quantile(n1, .95)), max=float(n1.max()),
                                                p_single_power=float((1 + (n1 >= C['models']['single_power']['r2_test']).sum()) / (1 + len(n1))),
                                                p_single_power_fixed=float((1 + (n1 >= C['models']['single_power_coef_2_over_beta']['r2_test']).sum()) / (1 + len(n1))))
    log(f'arm C models {json.dumps(clean({k_: (v["r2_test"], v.get("beta"), v["coef"]) for k_, v in C["models"].items()}))}; null5 {C["null_5_random_smooth"]}; null1 {C["null_1_random_smooth_best_of_66"]}')
    save()
    rb = np.random.default_rng(2746); BT = {nm: dict(r2=[], beta=[], coef=[]) for nm in list(MODELS) + ['single_power', 'single_power_coef_2_over_beta']}
    for _ in range(NB):
        a1 = agg('fit', rb.poisson(1.0, nf_).astype(float)); a2 = agg('test', rb.poisson(1.0, nt_).astype(float))
        for nm, fn in MODELS.items():
            co = E.ridge(fn(a1[1], a1[2]), a1[0], alph[nm]); BT[nm]['r2'].append(E.r2(a2[0], fn(a2[1], a2[2]) @ co)); BT[nm]['coef'].append(co)
        for nm, fixed in (('single_power', False), ('single_power_coef_2_over_beta', True)):
            gi, c, _ = single(a1[0], a1[1], fixed); BT[nm]['r2'].append(E.r2(a2[0], c * a2[1][:, gi])); BT[nm]['beta'].append(E.BGRID[gi]); BT[nm]['coef'].append([c])
    for nm in BT:
        C['models'][nm]['r2_test_ci'] = E.qci(BT[nm]['r2']); cb = np.array(BT[nm]['coef']); C['models'][nm]['coef_ci'] = [E.qci(cb[:, i]) for i in range(cb.shape[1])]
        if BT[nm]['beta']: C['models'][nm]['beta_ci'] = E.qci(BT[nm]['beta'])
    C['dr2_single_minus_M2_beta1_ci'] = E.qci(np.array(BT['single_power']['r2']) - np.array(BT['M2_beta1']['r2']))
    R['armC'] = C; R['seconds'] = time.time() - T0; save()
    log(f'done; {json.dumps(clean({k_: (v["r2_test"], v["r2_test_ci"], v.get("beta"), v.get("beta_ci")) for k_, v in C["models"].items()}))}')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--arm', choices=('AB', 'C'), required=True); ap.add_argument('--smoke', type=int, default=0); a = ap.parse_args()
    from interp.jedi_core import Jedi, sample
    from interp.x10_das_jedi import clean, holm, mass, RidgeR
    from interp.h12_boost_panel import mcnemar
    E.OUT.mkdir(parents=True, exist_ok=True); net = Jedi(64)
    if a.arm == 'AB': arm_AB(a, net, sample, clean, holm, mass, RidgeR, mcnemar)
    else: arm_C(a, net, sample, clean, holm, mass)


if __name__ == '__main__':
    main()
