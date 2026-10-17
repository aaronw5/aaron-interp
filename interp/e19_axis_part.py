"""E19 — ParT_full / ParT_kin: per true class, the mass read-out change under the WTA / hardest-particle axis vs the ½Δlog(M0·M2)
prediction, class flows, the Tbqq read-out distribution, the t→bqq' logit neurons, the three-prong prediction, and (ParT_full)
the axis change with the non-kinematic inputs fed as 0.

Sample: rows 8,000–9,000 of each class's first test file (10,000 jets; interp.p1v_verify_part.prep). Axes as P1v check 4
(interp/p1v_verify_part.axis_set, reaxis): stored (identity), escheme (control), wta, hardest; 4-vectors unchanged, only deta,
dphi, deltaR recomputed.
Read-out: ridge from the class token (128) to log m (m = invariant mass of the stored particles, ≥ 1 GeV), fitted on unedited
jets of the train half (P1's split function, seed 1); read-out numbers on the test half.
Matched calibration edit: every jet's (Δη, Δφ) scaled about the stored axis by k_j = exp(pred_j) and the 4-vectors rebuilt
(part_core.build): a real mass change of the predicted size.
Tbqq-logit neurons: W[:, Tbqq] · sd(z_k) on unedited jets (W = the last layer), 3 largest |contribution|.
Three-prong: Tbqq jets of the test half, exclusive kT 3 subjets, resolved as in interp/e19_common.py.
Input-group variants (ParT_full, wta axis and stored axis alike): (i) d0, d0err, dz, dzerr fed as 0; (ii) charge and the 5 PID
flags fed as 0; flows relative to the same variant on the stored axis.
  ./part_job.sh e19-part-smoke 'python -m interp.e19_axis_part 50'
  ./part_job.sh e19-part 'python -m interp.e19_axis_part'     -> results/part/e19_axis_by_class.json, .npz
"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.part_core import ParTNetwork, inputs, kin, build, moments, jmass, OUT, CL
from interp.p1v_verify_part import prep, split, Ridge, r2, fwd, axis_set, reaxis, wta_selftest
from jetdistill.part.network import FEATURES
from interp import e19_common as C

log = lambda *a: print(*a, flush=True)
iT, iH = CL.index('Tbqq'), CL.index('Hgg')
AXES = ('stored', 'escheme', 'wta', 'hardest')
MEDGES = np.array([0, 20, 40, 60, 75, 80, 91, 100, 110, 120, 130, 143, 155, 170, 190, 220, 260, 320, 2000.])
DISP = ('d0', 'd0err', 'dz', 'dzerr'); PIDF = ('charge', 'isChargedHadron', 'isNeutralHadron', 'isPhoton', 'isElectron', 'isMuon')


def stored_coords(S, ea, pa):
    J = S['Jb']; sgn = np.where(J['jet_eta'] < 0, -1.0, 1.0)
    return sgn * (ea - J['jet_eta'].astype(np.float64)), (pa - J['jet_phi'].astype(np.float64) + np.pi) % (2 * np.pi) - np.pi


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(82)
    PER = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    S = prep(PER, 8000); N = len(S['y']); y = S['y']; trf, trv, tr, te = split(N); te_mask = np.zeros(N, bool); te_mask[te] = True
    res = dict(per_class=PER, rows=[8000, 8000 + PER], N=N, classes=list(CL), wta_selftest=wta_selftest(), zmin=C.ZMIN, rmin=C.RMIN)
    log(f'E19 ParT: {N} jets rows 8000-{8000 + PER}; labels {np.bincount(y, minlength=10).tolist()} ({time.time() - t0:.0f} s)')
    A, ainfo = axis_set(S); res['axes_info'] = ainfo; AX = {}
    pt0, de0, dp0, ms0 = kin(S['Jb'])
    for a, (ea, pa) in A.items():
        E4, flips = reaxis(S['Jb'], ea, pa); M4 = moments(*kin(E4)[:3]); pred = 0.5 * np.log(M4['M0M2'] / S['M']['M0M2'])
        se, sp = stored_coords(S, ea, pa)
        Es = build(S['Jb'], pt0, de0 * np.exp(pred)[:, None], dp0 * np.exp(pred)[:, None], ms0)
        AX[a] = dict(E=E4, pred=pred, se=se, sp=sp, Es=Es, disp=np.hypot(se, sp),
                     scale_true_dlogm=np.log(np.maximum(jmass(Es), 1.0)) - S['logm0'])
        log(f'axis {a}: pred mean {pred.mean():+.4f}; disp median {np.median(AX[a]["disp"]):.4f}; sign flips {flips}')
    # three-prong (Tbqq, test half) for wta / hardest
    tt = te_mask & (y == iT); idx = np.flatnonzero(tt); P3 = {}
    for a in ('wta', 'hardest'):
        t1 = time.time(); P3[a] = C.three_prong(pt0[idx], de0[idx], dp0[idx], AX[a]['se'][idx], AX[a]['sp'][idx])
        log(f'three-prong {a}: {int(P3[a]["resolved"].sum())}/{len(idx)} resolved ({time.time() - t1:.0f} s)')
    res['models'] = {}; NPZ = dict(y=y, te=te_mask, m0=S['m0'], t_idx=idx)
    for a in AXES: NPZ[f'{a}_pred'] = AX[a]['pred']
    for a in ('wta', 'hardest'): NPZ[f'{a}_t_pred3'] = P3[a]['pred3']; NPZ[f'{a}_t_resolved'] = P3[a]['resolved']
    for n in ('full', 'kin'):
        net = ParTNetwork(n); t1 = time.time(); names = [f[0] for f in FEATURES[n]]
        xb, vb, mb = inputs(S['Jb'], n); z0, L0 = fwd(net, xb, vb, mb); c0 = L0.argmax(1)
        ridge = Ridge(z0, S['logm0'], trf, trv, tr); lm0 = ridge(z0); mh0 = np.exp(lm0)
        W = net.last[0]; sd = z0.std(0); contrib = W[:, iT] * sd; top3 = [int(i) for i in np.argsort(-np.abs(contrib))[:3]]
        row = dict(baseline=dict(acc=float((c0 == y).mean()), own=[float((c0[y == k] == k).mean()) for k in range(10)],
                                 ridge_r2_te=r2(lm0[te], S['logm0'][te]), alpha=float(ridge.alpha),
                                 ridge_r2_te_class=[r2(lm0[te_mask & (y == k)], S['logm0'][te_mask & (y == k)]) for k in range(10)]),
                   t_neurons=dict(top3=top3, contrib=[float(contrib[k]) for k in top3], W_t=[float(W[k, iT]) for k in top3],
                                  sd=[float(sd[k]) for k in top3], contrib_abs_sum=float(np.abs(contrib).sum())),
                   curves_unedited=dict(edges=MEDGES.tolist(), all=C.curve(mh0[te], z0[te][:, top3], MEDGES),
                                        t_jets=C.curve(mh0[tt], z0[tt][:, top3], MEDGES)), axes={})
        NPZ[f'{n}_mhat0'] = mh0; NPZ[f'{n}_c0'] = c0
        log(f'ParT_{n}: baseline {row["baseline"]}; Tbqq neurons {top3} contrib {np.round(row["t_neurons"]["contrib"], 3).tolist()}')
        dls_all = {}
        for a in AXES:
            x4, v4, m4 = inputs(AX[a]['E'], n); assert np.array_equal(m4, mb)
            z, L = fwd(net, x4, v4, m4); c = L.argmax(1); lm = ridge(z); dl = lm - lm0
            xs, vs, ms_ = inputs(AX[a]['Es'], n); zs, _ = fwd(net, xs, vs, ms_); dls = ridge(zs) - lm0; dls_all[a] = dls
            e = dict(acc=float((c == y).mean()), changed=float((c != c0).mean()),
                     disp_median_class=[float(np.median(AX[a]['disp'][y == k])) for k in range(10)],
                     scale_true_dlogm_class=[float(AX[a]['scale_true_dlogm'][te_mask & (y == k)].mean()) for k in range(10)], classes={})
            for k in range(10):
                e['classes'][CL[k]] = C.class_block(dl, AX[a]['pred'], dls, c0, c, y, k, te_mask, rng, 10)
            mh1 = np.exp(lm); ta = y == iT
            tj = dict(mhat0_q=np.quantile(mh0[tt], [.1, .25, .5, .75, .9]).tolist(), mhat1_q=np.quantile(mh1[tt], [.1, .25, .5, .75, .9]).tolist(),
                      frac_above_143_0=float((mh0[tt] > 143).mean()), frac_above_143_1=float((mh1[tt] > 143).mean()),
                      neurons={f'h{k}': dict(mean0=float(z0[ta, k].mean()), mean1=float(z[ta, k].mean()), dmean_ci=C.boot_mean(z[ta, k] - z0[ta, k], rng),
                                             sd=float(sd[k])) for k in top3},
                      t_logit_dmean=float((L[ta, iT] - L0[ta, iT]).mean()), t_logit_dmean_ci=C.boot_mean(L[ta, iT] - L0[ta, iT], rng))
            if a in ('wta', 'hardest'):
                rs = P3[a]['resolved']; d = dl[idx]; pr = AX[a]['pred'][idx]; p3 = P3[a]['pred3']; pp = P3[a]['pred3pp']
                tj['three_prong'] = dict(n=int(len(idx)), n_resolved=int(rs.sum()),
                                         axis_subjet_rank_frac=[float((P3[a]['axis_subjet'][rs] == s).mean()) for s in range(3)],
                                         axis_subjet_z_median=float(np.median(P3[a]['axis_subjet_z'][rs])),
                                         reading_factor_pred3_median=float(np.median(np.exp(p3[rs]))), reading_factor_pp_median=float(np.nanmedian(np.exp(pp[rs]))),
                                         pred3_mean=float(np.mean(p3[rs])), pred_actual_mean=float(pr[rs].mean()), dl_mean=float(d[rs].mean()),
                                         dl_vs_pred3=C.fit_line(p3[rs], d[rs]), dl_vs_pred3pp=C.fit_line(pp[rs], d[rs]),
                                         dl_vs_pred_actual=C.fit_line(pr[rs], d[rs]), pred_actual_vs_pred3=C.fit_line(p3[rs], pr[rs]))
            e['top'] = tj
            if a == 'wta':
                e['curves_edited'] = dict(all=C.curve(mh1[te], z[te][:, top3], MEDGES), t_jets=C.curve(mh1[tt], z[tt][:, top3], MEDGES))
            NPZ[f'{n}_{a}_mhat'] = mh1; NPZ[f'{n}_{a}_c'] = c
            row['axes'][a] = e
            log(f'  ParT_{n} {a}: acc {e["acc"]:.4f} changed {e["changed"]:.4f}; ' + '; '.join(
                f'{CL[k]} dl {e["classes"][CL[k]]["dlogm_hat"]:+.3f}/pred {e["classes"][CL[k]]["pred"]:+.3f}/cal {e["classes"][CL[k]]["dlogm_hat_cal"]:+.3f} '
                f'chg {e["classes"][CL[k]]["changed"]:.3f}' for k in range(10)))
            log(f'    top: {json.dumps({k: v for k, v in tj.items()})}')
        # control comparison + Holm
        ps = []
        for a in ('wta', 'hardest'):
            for k in range(10):
                s = te_mask & (y == k); d = np.log(NPZ[f'{n}_{a}_mhat'][s]) - np.log(NPZ[f'{n}_escheme_mhat'][s])
                row['axes'][a]['classes'][CL[k]]['vs_control'] = dict(diff=float(d.mean()), ci=C.boot_mean(d, rng), test=C.ttest1(d))
                ps.append(row['axes'][a]['classes'][CL[k]]['H0_readout_zero']['p'])
        hp = C.holm(ps); i = 0
        for a in ('wta', 'hardest'):
            for k in range(10): row['axes'][a]['classes'][CL[k]]['H0_holm_p'] = hp[i]; i += 1
        # input-group variants (ParT_full)
        if n == 'full':
            row['variants'] = {}
            xw, vw, mw = inputs(AX['wta']['E'], n)
            for vn, grp in (('disp_zero', DISP), ('pid_zero', PIDF)):
                ii = [names.index(k) for k in grp]
                xb_ = xb.copy(); xb_[:, ii] = 0; xw_ = xw.copy(); xw_[:, ii] = 0
                _, Lb = fwd(net, xb_, vb, mb); _, Lw = fwd(net, xw_, vw, mw); cb, cw = Lb.argmax(1), Lw.argmax(1)
                v = dict(acc_stored=float((cb == y).mean()), acc_wta=float((cw == y).mean()), classes={})
                for k in range(10):
                    s = y == k; ch = int((cw[s] != cb[s]).sum())
                    v['classes'][CL[k]] = dict(own_stored=float((cb[s] == k).mean()), own_wta=float((cw[s] == k).mean()),
                                               changed=ch / s.sum(), changed_ci=C.wilson(ch, int(s.sum())),
                                               new_call_frac=[float((cw[s] == j).mean()) for j in range(10)],
                                               mcnemar_own=C.mcnemar(cb[s] == k, cw[s] == k))
                row['variants'][vn] = v
                log(f'  variant {vn}: acc stored {v["acc_stored"]:.4f} wta {v["acc_wta"]:.4f}; Tbqq own {v["classes"]["Tbqq"]["own_stored"]:.3f} -> '
                    f'{v["classes"]["Tbqq"]["own_wta"]:.3f}; Hgg {v["classes"]["Hgg"]["own_stored"]:.3f} -> {v["classes"]["Hgg"]["own_wta"]:.3f}')
        res['models'][n] = row; log(f'ParT_{n} done ({time.time() - t1:.0f} s)'); del net
    sfx = '' if PER == 1000 else f'_{PER}'
    json.dump(res, open(OUT / f'e19_axis_by_class{sfx}.json', 'w'), indent=1)
    np.savez_compressed(OUT / f'e19_axis_by_class{sfx}.npz', **NPZ)
    log(f'E19 ParT DONE ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
