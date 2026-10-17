"""E19 — JEDI-linear (3-feature, n64): per true class, the mass read-out change under the WTA / hardest-particle axis vs the
½Δlog(M0·M2) prediction, the class flows, and for top jets the t-logit neurons and the three-prong prediction.

Sample: sample(per_class=4000, seed=81) (20,000 test jets). Edits as E17 (interp/e17_axis_jedi.py: reaxis, axis_set): only the
real particles' (Δη, Δφ) are recomputed about the new axis; pT and padding unchanged. Axes: stored (identity), centroid
(control), wta, hardest.
Read-out: ridge from z (the 16 pre-ReLU neurons of the last hidden layer) to log m, fitted on the unedited jets of the train
half (50/50 split, rng seed 81; alpha on the last quarter of the train half); all read-out numbers on the test half.
Matched calibration edit (per axis): every jet's angles scaled about the stored axis by k_j = exp(pred_j), pred_j = that jet's
½Δlog(M0·M2) under the axis change; a real mass change of the predicted size. ratio_cal = mean Δlog m̂(axis) / mean Δlog m̂(scale).
t-logit neurons: contribution of z_k to the t logit = K7[k, t] · sd(ReLU(z_k)) on unedited jets (K7 = the fused last-layer kernel);
the 3 largest |contribution|. Their response curves vs read-out mass on unedited test-half jets (and on WTA-edited jets).
Three-prong: top jets of the test half, exclusive kT 3 subjets (interp/e19_common.py), resolved if every subjet has pT fraction
≥ 0.10 and every pairwise ΔR ≥ 0.10.
  ./job.sh e19-jedi-smoke 'python -m interp.e19_axis_jedi 100'
  ./job.sh e19-jedi 'python -m interp.e19_axis_jedi'      -> results/jedi/e19_axis_by_class.json, e19_axis_by_class.npz
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp.e17_axis_jedi import axis_set, reaxis, moments, Ridge, r2, wta_selftest
from interp import e19_common as C

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'
CL = ['g', 'q', 'W', 'Z', 't']; T = 4
SEED = 81
AXES = ('stored', 'centroid', 'wta', 'hardest')
MEDGES = np.array([0, 20, 40, 60, 75, 80, 91, 100, 110, 120, 130, 143, 155, 170, 190, 220, 260, 320, 2000.])
log = lambda *a: print(*a, flush=True)


def scale_edit(x, k):
    real = x[..., 0] > 0; e = x.copy()
    for i in (1, 2):
        e[..., i] = np.where(real, x[..., i].astype(np.float64) * k[:, None], x[..., i]).astype(np.float32)
    return e


def main(PER):
    from interp.jedi_core import Jedi, sample
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(SEED + 1)
    res = dict(per_class=PER, seed=SEED, classes=CL, wta_selftest=wta_selftest(), zmin=C.ZMIN, rmin=C.RMIN, nboot=C.NBOOT)
    x, y, _ = sample(per_class=PER, seed=SEED); x = np.ascontiguousarray(x, np.float32); J = len(x); res['N'] = J
    net = Jedi(64); log(f'E19 JEDI: {J} jets, seed {SEED}; {net.path}')
    M = moments(x)
    R0 = net.run(x, keep=('z', 'logits')); z0, L0 = R0['z'], R0['logits']; c0 = L0.argmax(1)
    p = np.random.default_rng(SEED).permutation(J); tr, te = np.sort(p[:J // 2]), np.sort(p[J // 2:])
    trf, trv = tr[: len(tr) * 3 // 4], tr[len(tr) * 3 // 4:]
    ridge = Ridge(z0, M['logm'], trf, trv, tr); lm0 = ridge(z0); te_mask = np.zeros(J, bool); te_mask[te] = True
    res['baseline'] = dict(acc=float((c0 == y).mean()), own=[float((c0[y == k] == k).mean()) for k in range(5)],
                           ridge_r2_te=r2(lm0[te], M['logm'][te]), ridge_alpha=float(ridge.alpha),
                           ridge_r2_te_class=[r2(lm0[te][y[te] == k], M['logm'][te][y[te] == k]) for k in range(5)],
                           readout_bias_class=[float((lm0 - M['logm'])[te_mask & (y == k)].mean()) for k in range(5)])
    log(f'baseline {res["baseline"]}')
    # ---- t-logit neurons
    K7 = np.squeeze(np.asarray(net.KB['d7'][0], np.float64)); K7 = K7.T if K7.shape == (5, 16) else K7; assert K7.shape == (16, 5), K7.shape; rz0 = np.maximum(z0, 0); sd = rz0.std(0)
    contrib = K7[:, T] * sd; top3 = [int(i) for i in np.argsort(-np.abs(contrib))[:3]]
    res['t_neurons'] = dict(K7_t=K7[:, T].tolist(), sd_relu=sd.tolist(), contrib=contrib.tolist(), top3=top3,
                            K7_all_top3=[K7[k].tolist() for k in top3],
                            on_frac_all=[float((z0[:, k] > 0).mean()) for k in top3])
    log(f't-logit contributions K7[:,t]·sd: {np.round(contrib, 3).tolist()}; top3 z{top3}')
    A, ainfo = axis_set(x); res['axes_info'] = ainfo
    mh0 = np.exp(lm0)
    res['curves_unedited'] = dict(edges=MEDGES.tolist(), all=C.curve(mh0[te], z0[te][:, top3], MEDGES),
                                  t_jets=C.curve(mh0[te_mask & (y == T)], z0[te_mask & (y == T)][:, top3], MEDGES),
                                  t_logit_margin=C.curve(mh0[te], (L0[te, T] - np.delete(L0[te], T, 1).max(1))[:, None], MEDGES),
                                  true_mass_all=C.curve(M['m'][te], z0[te][:, top3], MEDGES))
    res['axes'] = {}; NPZ = dict(y=y, te=te_mask, m=M['m'], mhat0=mh0, c0=c0, z0_top3=z0[:, top3])
    for a in AXES:
        ae, ap = A[a]; xa = reaxis(x, ae, ap); Ma = moments(xa)
        pred = 0.5 * np.log((Ma['M0'] * Ma['M2']) / (M['M0'] * M['M2']))
        Ra = net.run(xa, keep=('z', 'logits')); z, L = Ra['z'], Ra['logits']; c = L.argmax(1); lm = ridge(z); dl = lm - lm0
        xs = scale_edit(x, np.exp(pred)); Ms = moments(xs); Rs = net.run(xs, keep=('z',)); dls = ridge(Rs['z']) - lm0
        e = dict(disp_median_class=[float(np.median(np.hypot(ae, ap)[y == k])) for k in range(5)],
                 acc=float((c == y).mean()), changed=float((c != c0).mean()),
                 scale_true_dlogm_class=[float((Ms['logm'] - M['logm'])[te_mask & (y == k)].mean()) for k in range(5)],
                 classes={})
        for k in range(5):
            e['classes'][CL[k]] = C.class_block(dl, pred, dls, c0, c, y, k, te_mask, rng, 5)
            # control comparison filled below
        # top jets: neurons, threshold, three-prong
        tt = te_mask & (y == T); ta = y == T
        mh1 = np.exp(lm)
        tj = dict(mhat0_q=np.quantile(mh0[tt], [.1, .25, .5, .75, .9]).tolist(), mhat1_q=np.quantile(mh1[tt], [.1, .25, .5, .75, .9]).tolist(),
                  frac_above_143_0=float((mh0[tt] > 143).mean()), frac_above_143_1=float((mh1[tt] > 143).mean()),
                  frac_above_143_ci0=C.wilson(int((mh0[tt] > 143).sum()), int(tt.sum())), frac_above_143_ci1=C.wilson(int((mh1[tt] > 143).sum()), int(tt.sum())),
                  mcnemar_above_143=C.mcnemar(mh0[tt] > 143, mh1[tt] > 143), neurons={})
        for k in top3:
            on0, on1 = z0[ta, k] > 0, z[ta, k] > 0
            tj['neurons'][f'z{k}'] = dict(on0=float(on0.mean()), on1=float(on1.mean()), on0_ci=C.wilson(int(on0.sum()), int(ta.sum())),
                                          on1_ci=C.wilson(int(on1.sum()), int(ta.sum())), mcnemar=C.mcnemar(on0, on1),
                                          mean0=float(z0[ta, k].mean()), mean1=float(z[ta, k].mean()),
                                          dmean_ci=C.boot_mean(z[ta, k] - z0[ta, k], rng),
                                          on0_W=float((z0[y == 2, k] > 0).mean()), on1_W=float((z[y == 2, k] > 0).mean()),
                                          on0_Z=float((z0[y == 3, k] > 0).mean()), on1_Z=float((z[y == 3, k] > 0).mean()))
        mg0 = L0[:, T] - np.delete(L0, T, 1).max(1); mg1 = L[:, T] - np.delete(L, T, 1).max(1)
        tj['t_margin_mean0'] = float(mg0[ta].mean()); tj['t_margin_mean1'] = float(mg1[ta].mean())
        tj['t_logit_dmean'] = float((L[ta, T] - L0[ta, T]).mean()); tj['t_logit_dmean_ci'] = C.boot_mean(L[ta, T] - L0[ta, T], rng)
        if a in ('wta', 'hardest'):
            t1 = time.time(); idx = np.flatnonzero(tt)
            P3 = C.three_prong(x[idx, :, 0], x[idx, :, 1], x[idx, :, 2], ae[idx], ap[idx]); rs = P3['resolved']
            tj['three_prong'] = dict(n=int(len(idx)), n_resolved=int(rs.sum()), seconds=time.time() - t1,
                                     axis_subjet_rank_frac=[float((P3['axis_subjet'][rs] == s).mean()) for s in range(3)],
                                     axis_subjet_z_median=float(np.median(P3['axis_subjet_z'][rs])),
                                     reading_factor_pred3_median=float(np.median(np.exp(P3['pred3'][rs]))),
                                     reading_factor_pp_median=float(np.median(np.exp(P3['pred3pp'][rs]))),
                                     pred3_mean=float(np.nanmean(P3['pred3'][rs])), pred3pp_mean=float(np.nanmean(P3['pred3pp'][rs])),
                                     pred_actual_mean=float(pred[idx][rs].mean()), dl_mean=float(dl[idx][rs].mean()),
                                     dl_vs_pred3=C.fit_line(P3['pred3'][rs], dl[idx][rs]), dl_vs_pred3pp=C.fit_line(P3['pred3pp'][rs], dl[idx][rs]),
                                     dl_vs_pred_actual=C.fit_line(pred[idx][rs], dl[idx][rs]),
                                     pred_actual_vs_pred3=C.fit_line(P3['pred3'][rs], pred[idx][rs]),
                                     unresolved_dl_mean=float(dl[idx][~rs].mean()) if (~rs).any() else None,
                                     unresolved_pred_mean=float(pred[idx][~rs].mean()) if (~rs).any() else None,
                                     dl_mean_axis_on_hardest_subjet=float(dl[idx][rs & (P3['axis_subjet'] == 0)].mean()) if (rs & (P3['axis_subjet'] == 0)).any() else None,
                                     dl_mean_axis_on_other_subjet=float(dl[idx][rs & (P3['axis_subjet'] > 0)].mean()) if (rs & (P3['axis_subjet'] > 0)).any() else None)
            NPZ[f'{a}_t_pred3'] = P3['pred3']; NPZ[f'{a}_t_resolved'] = rs; NPZ[f'{a}_t_idx'] = idx
            log(f'  {a}: three-prong {tj["three_prong"]}')
        e['top'] = tj
        if a == 'wta':
            e['curves_edited'] = dict(all=C.curve(mh1[te], z[te][:, top3], MEDGES), t_jets=C.curve(mh1[tt], z[tt][:, top3], MEDGES))
        NPZ[f'{a}_mhat'] = mh1; NPZ[f'{a}_c'] = c; NPZ[f'{a}_pred'] = pred; NPZ[f'{a}_dl_scale'] = dls; NPZ[f'{a}_z_top3'] = z[:, top3]
        res['axes'][a] = e
        log(f'axis {a}: acc {e["acc"]:.4f} changed {e["changed"]:.4f}; ' + '; '.join(
            f'{CL[k]}: dl {e["classes"][CL[k]]["dlogm_hat"]:+.4f} pred {e["classes"][CL[k]]["pred"]:+.4f} cal {e["classes"][CL[k]]["dlogm_hat_cal"]:+.4f} '
            f'chg {e["classes"][CL[k]]["changed"]:.3f}' for k in range(5)) + f' ({time.time() - t0:.0f} s)')
        log(f'   top: {json.dumps({k: v for k, v in tj.items() if k != "three_prong"})}')
    # control comparison (paired: axis vs centroid read-out change, per class, test half) and Holm over class × {wta, hardest}
    dlc = NPZ['centroid_mhat']
    ps = []
    for a in ('wta', 'hardest'):
        for k in range(5):
            s = te_mask & (y == k); d = np.log(NPZ[f'{a}_mhat'][s]) - np.log(dlc[s])
            res['axes'][a]['classes'][CL[k]]['vs_control'] = dict(diff=float(d.mean()), ci=C.boot_mean(d, rng), test=C.ttest1(d))
            ps.append(res['axes'][a]['classes'][CL[k]]['H0_readout_zero']['p'])
    hp = C.holm(ps); i = 0
    for a in ('wta', 'hardest'):
        for k in range(5): res['axes'][a]['classes'][CL[k]]['H0_holm_p'] = hp[i]; i += 1
    sfx = '' if PER == 4000 else f'_{PER}'
    json.dump(res, open(OUT / f'e19_axis_by_class{sfx}.json', 'w'), indent=1)
    np.savez_compressed(OUT / f'e19_axis_by_class{sfx}.npz', **NPZ)
    log(f'wrote {OUT}/e19_axis_by_class{sfx}.json; E19 JEDI DONE ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4000)
