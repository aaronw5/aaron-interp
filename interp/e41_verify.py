"""E41 verification (fresh samples; key parts only) plus the missing mass-preserving control.

Network: official JEDI-linear 3-feature n64 (exact forward pass, interp.jedi_core.Jedi / x10_das_jedi.JNet).
Samples: fit = sample(per_class=PC, seed=S1), test = sample(PC, S2) minus jets also in fit (default 500, 4121 / 4122).
Pool: E41.build_pool (nat, shift3/6, turn_hard14 2/5/10, turn_soft, turnup, dil2/5/10), unchanged.
New edit 'ctrl10' (E27w control, mass preserving): the same 14 hardest particles turned by the same azimuth amounts |dtheta_i|
  as the 10 % turn, each sign random, then a minimum-norm azimuth correction on the same particles restoring the vector M1, with a
  common rescale holding the count-weighted displacement D equal to the turn's (E27w.solve_ctrl).  pT, r, M0, M2, m_A, M1 and so
  m are held; only the azimuthal arrangement changes.  Kept where turn_hard14_10 is kept and the control converges.
Parts:
 1. calls: share of calls changed and W->Z / Z->W flow for turn10, ctrl10, dil10 on the same jets; McNemar turn vs ctrl.
 2. read-outs (all 10 stages; ridge + MLP 64 hidden, fit on all fit-sample pool rows, as E41 'pool'): R^2 and corr of Delta on the
    test turned subset and shifted subset; mean change of the Delta read-out under turn10 and under ctrl10 (same jets) vs the
    true Delta change; R_m of the ridge log-m probe fitted on nat+shift+dil for turn10 and for ctrl10 (response / dil10 response).
    Same for 2 untrained redraws and 2 row-permuted copies (mini band).
 3. interchange at S/h4/h5 (exact tail): 1-dim DAS trained on fit-sample (turned, original) pairs toward the original's call
    (v_turn) and on (ctrl-edited, original) pairs toward the original's call (v_ctrl); each scored on both test pair sets;
    100 random unit directions as null.
Output: $AI_RESULTS/jedi/e41_verify.json   python -m interp.e41_verify [--seeds 4121,4122] [--pc 500] [--smoke]
"""
import argparse
import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'
T0 = time.time(); W_, Z_ = 2, 3
warnings.filterwarnings('ignore')


def log(*a): print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def ctrl_rows(x, P, seed):
    from interp.e27w_mass_vs_m2_jedi import solve_B, solve_ctrl, M1v, to_x, wrap, disp
    x64 = np.asarray(x, np.float64); pt = x64[..., 0]; real = pt > 0; r = np.hypot(x64[..., 1], x64[..., 2]); th = np.arctan2(x64[..., 2], x64[..., 1])
    rank = np.argsort(np.argsort(-np.where(real, pt, -1), 1), 1); sub = real & (rank < 14)
    M10 = M1v(pt, r, th); phu = np.arctan2(M10[:, 1], M10[:, 0]); d = np.where(sub, wrap(phu[:, None] - th), 0.0)
    tB, ok = solve_B(pt, r, th, d, 0.10); thB = th + tB[:, None] * d; xB = to_x(x, real, r, thB)
    kT = P['rowidx']['turn_hard1410'] >= 0
    same = float(np.abs(xB[kT] - P['x'][P['rowidx']['turn_hard1410'][kT]]).max()) if kT.any() else None
    DB = disp(r, th, thB); thC, okC = solve_ctrl(pt, r, th, sub, tB[:, None] * d, DB, np.random.default_rng(seed))
    keep = kT & okC; xC = to_x(x, real, r, thC)
    return xC, keep, dict(turn_recompute_maxdiff=same, n_turn=int(kT.sum()), n_ctrl_kept=int(keep.sum()))


def main(a):
    import interp.e41_true_mass_layers_jedi as E
    from interp.x10_das_jedi import clean, JNet, das_fit, proj as qclip
    from interp.h12_boost_panel import mcnemar
    sf, st = a.seeds; pc = a.pc; NB = 200 if a.smoke else 1000; NR = 10 if a.smoke else 100; steps = 60 if a.smoke else 400
    mlp_n, mlp_it = (2000, 20) if a.smoke else (12000, 80)
    OUT.mkdir(parents=True, exist_ok=True); JS = OUT / ('e41_verify_smoke.json' if a.smoke else 'e41_verify.json')
    R = dict(node='E41-verify', seeds=[sf, st], per_class=pc); save = lambda: JS.write_text(json.dumps(clean(R), indent=1))  # noqa: E731
    xf, yf, xt, yt, Pf, Pt, ov = E.load_data(sf, st, pc, False)
    R['samples'] = dict(N_fit=len(yf), N_test=len(yt), overlap_dropped=ov, rows_fit=len(Pf['jid']), rows_test=len(Pt['jid']),
                        kept_test={k: int((v >= 0).sum()) for k, v in Pt['rowidx'].items()})
    xCf, kCf, If = ctrl_rows(xf, Pf, 9001); xCt, kCt, It = ctrl_rows(xt, Pt, 9002)
    iT, iD = Pt['rowidx']['turn_hard1410'], Pt['rowidx']['dil10']; J = np.flatnonzero(kCt & (iD >= 0))   # matched test jets
    obT = {k: v[iT[J]] for k, v in Pt['obs'].items()}; obD = {k: v[iD[J]] for k, v in Pt['obs'].items()}
    oC = E.jet_obs(xCt[J]); o0 = E.jet_obs(xt[J])
    R['ctrl_edit'] = dict(fit=If, test=It, n_matched_test=len(J),
                          median_dlogm=dict(turn=float(np.median(obT['logm'] - o0['logm'])), ctrl=float(np.median(oC['logm'] - o0['logm'])), dil=float(np.median(obD['logm'] - o0['logm']))),
                          max_abs_dlogmA_ctrl=float(np.abs(oC['logmA'] - o0['logmA']).max()), max_abs_dlogm_ctrl=float(np.abs(oC['logm'] - o0['logm']).max()),
                          median_M1_over_M0=dict(nat=float(np.median(o0['M1'] / o0['M0'])), turn=float(np.median(obT['M1'] / obT['M0'])), ctrl=float(np.median(oC['M1'] / oC['M0']))))
    log('ctrl edit:', json.dumps(clean(R['ctrl_edit']))); save()
    Wb = np.random.default_rng(4141).poisson(1.0, (NB, len(yt))).astype(np.float64); wJ = Wb[:, J]

    # ---------------------------------------------------------------- 1 + 2: trained and mini band
    nets = ['trained'] + ([] if a.smoke else ['u0', 'u1', 'p0', 'p1']); R['readouts'] = {}
    for nm in nets:
        t = time.time(); run, desc = E.runner(nm)
        Ff = E.feats(run, Pf['x']); Ft = E.feats(run, Pt['x']); FC = E.feats(run, xCt[J])
        natt = Pt['fam'] == 'nat'; acc = float((Ft['logits'][natt].argmax(1) == yt[Pt['jid'][natt]]).mean()); o = dict(desc=desc, acc_test=acc)
        if nm == 'trained':
            c0 = Ft['logits'][Pt['rowidx']['nat'][J]].argmax(1); cT = Ft['logits'][iT[J]].argmax(1); cD = Ft['logits'][iD[J]].argmax(1); cC = FC['logits'].argmax(1)
            def fl(c):
                ch = c != c0; return dict(changed=float(ch.mean()), changed_ci=E.qci((wJ @ ch.astype(float)) / wJ.sum(1)),
                                          WZ=float((c[c0 == W_] == Z_).mean()), ZW=float((c[c0 == Z_] == W_).mean()), n_W=int((c0 == W_).sum()), n_Z=int((c0 == Z_).sum()))
            R['calls'] = dict(n=len(J), turn10=fl(cT), ctrl10=fl(cC), dil10=fl(cD), mcnemar_turn_vs_ctrl=mcnemar(cT != c0, cC != c0),
                              mcnemar_turn_vs_dil=mcnemar(cT != c0, cD != c0))
            log('calls:', json.dumps(clean(R['calls'])))
        Sf, St = E.fit_sets(Pf), E.fit_sets(Pt); ST = {}
        for stg in E.STG:
            e = {}; yD_f, yD_t = Pf['obs']['Delta'], Pt['obs']['Delta']
            for kind in ('ridge', 'mlp'):
                pr = E.Ridge(Ff[stg], yD_f, Pf['jid']) if kind == 'ridge' else E.MLP(Ff[stg], yD_f, mlp_n, 41, mlp_it)
                p = pr(Ft[stg]); pc_ = pr(FC[stg]); m = St['turn']; ms = St['shift']
                d0 = p[Pt['rowidx']['nat'][J]]; dT = p[iT[J]] - d0; dC = pc_ - d0; tT = yD_t[iT[J]] - yD_t[Pt['rowidx']['nat'][J]]
                e[kind] = dict(on_turn=E.r2_boot(yD_t[m], p[m], Pt['jid'][m], Wb), on_shift=E.r2_boot(yD_t[ms], p[ms], Pt['jid'][ms], Wb),
                               mean_change_turn10=float(dT.mean()), mean_change_ctrl10=float(dC.mean()), true_change_turn10=float(tT.mean()),
                               slope_turn=float(dT.mean() / tT.mean()), slope_turn_ci=E.qci((wJ @ dT) / (wJ @ tT)),
                               ctrl_over_turn=float(dC.mean() / dT.mean()) if dT.mean() != 0 else None,
                               ctrl_over_turn_ci=E.qci((wJ @ dC) / np.where(np.abs(wJ @ dT) > 0, wJ @ dT, np.nan)))
            m1 = Sf['nsd']; pr1 = E.Ridge(Ff[stg][m1], Pf['obs']['logm'][m1], Pf['jid'][m1]); pall = pr1(Ft[stg]); pC = pr1(FC[stg])
            a0 = pall[Pt['rowidx']['nat'][J]]; rT = pall[iT[J]] - a0; rD = pall[iD[J]] - a0; rC = pC - a0
            sgn = np.sign(rD.mean()) if rD.mean() != 0 else 1.0
            den = wJ @ rD; den = np.where(np.abs(den) > 0, den, np.nan)
            e['Rm_logm_nsd'] = dict(turn=float(rT.mean() / rD.mean()), turn_ci=E.qci((wJ @ rT) / den), ctrl=float(rC.mean() / rD.mean()),
                                    ctrl_ci=E.qci((wJ @ rC) / den), turn_minus_ctrl_ci=E.qci((wJ @ (rT - rC)) / den), dil_mean=float(rD.mean()), sign=float(sgn))
            ST[stg] = e
            log(f'  {nm} {stg}: Delta on turn R2 mlp {e["mlp"]["on_turn"]["r2"]:.3f} corr {e["mlp"]["on_turn"]["corr"]} ridge {e["ridge"]["on_turn"]["r2"]:.3f} corr {e["ridge"]["on_turn"]["corr"]};'
                f' shift R2 mlp {e["mlp"]["on_shift"]["r2"]:.3f}; Rm turn {e["Rm_logm_nsd"]["turn"]:.3f} ctrl {e["Rm_logm_nsd"]["ctrl"]:.3f}; mlp slope {e["mlp"]["slope_turn"]:.3f} ctrl/turn {e["mlp"]["ctrl_over_turn"]}')
        o['stages'] = ST; R['readouts'][nm] = o; save(); log(f'{nm} done ({time.time() - t:.0f} s), acc {acc:.4f}')
        del Ff, Ft, FC, run; gc.collect()

    # ---------------------------------------------------------------- 3: interchange
    jn = JNet(); keep = ('S', 'h4', 'h5', 'logits')
    def H(x): return jn.net.run(x, keep=keep, batch=1000)
    Jf = np.flatnonzero(kCf); iTf = Pf['rowidx']['turn_hard1410']
    Hf0, HfT, HfC = H(Pf['x'][Pf['rowidx']['nat'][Jf]]), H(Pf['x'][iTf[Jf]]), H(xCf[Jf])
    Ht0, HtT, HtC = H(Pt['x'][Pt['rowidx']['nat'][J]]), H(Pt['x'][iT[J]]), H(xCt[J])
    R['jnet_check'] = jn.check(Pt['x'][Pt['rowidx']['nat'][J]][:1500], {k: v[:1500] for k, v in Ht0.items()})
    cf = {k: v['logits'].argmax(1) for k, v in (('0', Hf0), ('T', HfT), ('C', HfC))}; ct = {k: v['logits'].argmax(1) for k, v in (('0', Ht0), ('T', HtT), ('C', HtC))}
    R['swap'] = dict(n_fit_pairs=len(Jf), n_test_pairs=len(J), n_informative_test=dict(turn=int((ct['T'] != ct['0']).sum()), ctrl=int((ct['C'] != ct['0']).sum())),
                     n_informative_fit=dict(turn=int((cf['T'] != cf['0']).sum()), ctrl=int((cf['C'] != cf['0']).sum())), res={})
    log('swap pairs:', json.dumps(R['swap']))

    def ev(stage, Q, hb, hs):
        return jn.exact(stage, qclip(jn.rng_[stage], hb, hb + ((hs - hb) @ Q) @ Q.T))[0].argmax(1)

    def summ(c, lab, basec, cR):
        inf = lab != basec; ok = (c == lab).astype(float); ch = (c != basec).astype(float); o = dict(n_inf=int(inf.sum()))
        if inf.any():
            o['IIA_inf'] = float(ok[inf].mean()); o['IIA_inf_ci'] = E.qci((wJ[:, inf] @ ok[inf]) / np.maximum(wJ[:, inf].sum(1), 1e-12))
            rI = (cR == lab[None])[:, inf].mean(1); o['random_IIA_inf_median'] = float(np.median(rI)); o['random_IIA_inf_q95'] = float(np.quantile(rI, .95))
            o['p_rank_IIA_inf'] = float((1 + (rI >= o['IIA_inf']).sum()) / (1 + len(rI)))
            o['excess_IIA_inf'] = o['IIA_inf'] - o['random_IIA_inf_median']
            o['excess_IIA_inf_boot'] = ((wJ[:, inf] @ ok[inf]) / np.maximum(wJ[:, inf].sum(1), 1e-12)) - o['random_IIA_inf_median']
        o['calls_changed'] = float(ch.mean()); rc = (cR != basec[None]).mean(1); o['random_calls_changed_median'] = float(np.median(rc))
        o['p_rank_calls_changed'] = float((1 + (rc >= o['calls_changed']).sum()) / (1 + len(rc)))
        bw, bz = basec == W_, basec == Z_; o['WZ'] = float((c[bw] == Z_).mean()) if bw.any() else None; o['ZW'] = float((c[bz] == W_).mean()) if bz.any() else None
        return o
    for stage in ('S', 'h4', 'h5'):
        t = time.time(); d = Ht0[stage].shape[1]; RD = np.random.default_rng(7 + d).standard_normal((NR, d)); RD /= np.linalg.norm(RD, axis=1, keepdims=True)
        perm = np.random.default_rng(4153).permutation(len(Jf)); Qs = {}
        for pn, Hb in (('turn', HfT), ('ctrl', HfC)):
            Qs[pn], info = das_fit(jn.tail(stage), Hb[stage][perm], Hf0[stage][perm], cf['0'][perm], cf['T' if pn == 'turn' else 'C'][perm], 1, 4154, steps=steps, lr=0.01, rg=jn.rng_[stage])
        o = dict(cos_vturn_vctrl=float(abs(Qs['turn'][:, 0] @ Qs['ctrl'][:, 0])))
        for pn, Hb, bk in (('turn', HtT, 'T'), ('ctrl', HtC, 'C')):
            hb, hs, lab, basec = Hb[stage], Ht0[stage], ct['0'], ct[bk]
            cR = np.stack([ev(stage, RD[i][:, None], hb, hs) for i in range(NR)])
            o[f'{pn}_pairs'] = {f'v_{q}': summ(ev(stage, Qs[q], hb, hs), lab, basec, cR) for q in ('turn', 'ctrl')}
        a_, b_ = o['turn_pairs']['v_turn'], o['ctrl_pairs']['v_ctrl']
        if 'excess_IIA_inf_boot' in a_ and 'excess_IIA_inf_boot' in b_:
            dd = a_['excess_IIA_inf_boot'] - b_['excess_IIA_inf_boot']
            o['excess_turn_minus_ctrl'] = dict(point=a_['excess_IIA_inf'] - b_['excess_IIA_inf'], ci=E.qci(dd), p_one_sided_le0=float((dd <= 0).mean()))
        for q in (a_, b_): q.pop('excess_IIA_inf_boot', None)
        for pn in ('turn_pairs', 'ctrl_pairs'):
            for q in o[pn].values(): q.pop('excess_IIA_inf_boot', None)
        R['swap']['res'][stage] = o; save()
        log(f'  {stage} ({time.time() - t:.0f} s): turn pairs v_turn IIA_inf {a_.get("IIA_inf")} (rand {a_.get("random_IIA_inf_median")}), v_ctrl {o["turn_pairs"]["v_ctrl"].get("IIA_inf")};'
            f' ctrl pairs v_ctrl IIA_inf {b_.get("IIA_inf")} (rand {b_.get("random_IIA_inf_median")}), v_turn {o["ctrl_pairs"]["v_turn"].get("IIA_inf")}; cos {o["cos_vturn_vctrl"]:.3f}; diff {o.get("excess_turn_minus_ctrl")}')
    R['seconds'] = time.time() - T0; save(); log(f'done, wrote {JS}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', default='4121,4122'); ap.add_argument('--pc', type=int, default=500)
    ap.add_argument('--smoke', action='store_true'); a = ap.parse_args(); a.seeds = tuple(int(s) for s in a.seeds.split(','))
    main(a)
