"""X9 verification (skeptic): rerun the edit block of interp/x9_efn_pfn (step 0, step 3a/3b/3c) on a fresh test sample for
JEDI-linear and the four selected flow models, plus two controls that X9 did not run.

  ./job.sh x9-verify 'python -m interp.x9_verify'

Fresh sample T' = sample(per_class=2000, seed=2027) (N = 10,000); fits (log m read-outs) on sample(6000, 0) minus T', 2/3 fit /
1/3 dev (permutation seed 0), as X9.  Edits (padding never moved), each run once for JEDI and for every model on the same jets:
  rotation (matched, X9 control) | translation 0.06 rad random direction / +eta / +phi | WTA axis | angle scaling k = exp(+mean
  pred) (calibration, real m increase) and k = exp(-mean pred) (real m decrease)
  NEW  M2-held translation: the random-direction translation followed by a per-jet angle scaling s = sqrt(M2 / M2') about the
       stored axis, so M0 and M2 (about the stored axis) are exactly as before, the jet is still displaced (|M1| changed) and the
       invariant mass changes by log s (about -0.3).  Predictions on W jets:
         H_M2   (the response to translation is carried by M2 about the stored axis): calls change about as for the rotation;
                W -> Z near the rotation level; mass read-out change ~ 0 (calibrated ratio ~ 0).
         H_m    (the network reads the invariant mass): read-out change ~ log s (calibrated ratio ~ -1, as the k = exp(-pred)
                scaling); W calls move to the low-mass classes (g, q) as under that scaling.
         H_off  (generic response to a displaced jet, not through M2): W -> Z / calls changed near the plain translation.
  NEW  rank swap with JEDI's units chosen as the flow models' are (the 10 S units of largest resample cost on T'), next to the
       X6 units [0, 43, 51, 52, 63] that X9 used, so the unit selection is matched.
Statistics: Wilson CIs for rates; 1000 jet bootstraps for read-out means and calibrated ratios; McNemar (exact) for paired call
changes; ratio CIs by bootstrap over common jets.  Writes $AI_RESULTS/jedi/x9_verify.json.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp.x9_efn_pfn import (Flow, moments, translate, rotate_matched, scale_angles, swap, Ridge, r2, boot, mcnemar, wilson,  # noqa: E402
                               indices, clean, OUT)

T0 = time.time(); W_, Z_ = 2, 3
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
MODELS = ['pfn_64-64-32_s0', 'pfn_64-64-32_s1', 'efn_100-100-128_s0', 'efn_100-100-128_s1']
SEED_T = int(os.environ.get('X9V_SEED', '2027'))


def main():
    from interp.jedi_core import Jedi, sample
    from interp.e17_axis_jedi import axis_set, reaxis
    JS = OUT / 'x9_verify.json'; R = dict(header='see interp/x9_verify.py docstring', models=MODELS)
    save = lambda: JS.write_text(json.dumps(clean(R), indent=1))
    xT, yT, _ = sample(2000, SEED_T); xF, yF, _ = sample(6000, 0)
    iT, iF = indices(2000, SEED_T), indices(6000, 0); keepF = ~np.isin(iF, iT)
    iT0 = indices(2000, 313)
    xF = np.ascontiguousarray(xF[keepF], np.float32); xT = np.ascontiguousarray(xT, np.float32); J = len(xT)
    perm = np.random.default_rng(0).permutation(len(xF)); nf = 2 * len(xF) // 3; fit, dev = np.sort(perm[:nf]), np.sort(perm[nf:])
    R['samples'] = dict(T=dict(per_class=2000, seed=SEED_T, N=J, overlap_with_X9_T=int(np.isin(iT, iT0).sum())),
                        F=dict(N=int(len(xF)), removed_overlap=int((~keepF).sum()), fit=int(len(fit)), dev=int(len(dev))))
    log('samples', R['samples'])
    isW = yT == W_; nW = int(isW.sum())
    MT = moments(xT); MF = moments(xF)
    # ---------------- edits (built once)
    E = {}
    E['rotation'] = rotate_matched(xT, 0.06, np.random.default_rng(66))
    th = np.random.default_rng(67).uniform(0, 2 * np.pi, J)
    E['transl_random'] = translate(xT, 0.06, th)
    E['transl_eta'] = translate(xT, 0.06, 0.0); E['transl_phi'] = translate(xT, 0.06, np.pi / 2)
    A, ainfo = axis_set(xT); E['wta'] = reaxis(xT, *A['wta'])
    Mq = moments(E['transl_random']); s = np.sqrt(MT['M2'] / np.maximum(Mq['M2'], 1e-30))
    E['transl_M2held'] = scale_angles(E['transl_random'], s)
    pred = {k: 0.5 * np.log(moments(v)['M2'] / MT['M2']) for k, v in E.items()}
    pred['wta'] = 0.5 * np.log(moments(E['wta'])['M0'] * moments(E['wta'])['M2'] / (MT['M0'] * MT['M2']))
    kc = float(np.exp(pred['transl_random'].mean()))
    E['scale_up'] = scale_angles(xT, np.full(J, kc)); E['scale_down'] = scale_angles(xT, np.full(J, 1 / kc))
    pred['scale_up'] = np.full(J, np.log(kc)); pred['scale_down'] = np.full(J, -np.log(kc))
    hyg = {}
    for k, v in E.items():
        Mv = moments(v); ok = MT['m'] > 1; real = xT[..., 0] > 0
        hyg[k] = dict(pred_half_dlogM2_mean=float(pred[k].mean()), true_dlogm_mean=float((Mv['logm'] - MT['logm'])[ok].mean()),
                      true_dlogm_maxabs=float(np.abs(Mv['logm'] - MT['logm'])[ok].max()),
                      dlogM0_maxabs=float(np.abs(np.log(Mv['M0'] / MT['M0'])).max()),
                      dlogM2_mean=float(np.log(Mv['M2'] / MT['M2']).mean()), dlogM2_maxabs=float(np.abs(np.log(Mv['M2'] / MT['M2'])).max()),
                      padding_changed=bool((v[~real] != xT[~real]).any()), pt_changed=bool((v[..., 0] != xT[..., 0]).any()),
                      centroid_offset_median=float(np.median(np.hypot((v[..., 0] * v[..., 1]).sum(1), (v[..., 0] * v[..., 2]).sum(1)) / Mv['M0'])))
    R['edit_hygiene'] = hyg; R['calibration_k'] = kc; R['wta_info'] = ainfo
    log('edit hygiene', json.dumps(clean(hyg))); save()

    def stats(c, c0, d, dcal, who_kind):
        out = dict(calls_changed=float((c != c0).mean()), calls_changed_ci=wilson(int((c != c0).sum()), J),
                   W_to_Z=float((c[isW] == Z_).mean()), W_to_Z_ci=wilson(int((c[isW] == Z_).sum()), nW),
                   W_to_gq=float(np.isin(c[isW], [0, 1]).mean()), W_to_gq_ci=wilson(int(np.isin(c[isW], [0, 1]).sum()), nW),
                   W_kept=float((c[isW] == W_).mean()))
        out['dlogm_hat'] = float(d.mean()); out['dlogm_hat_ci'] = boot(lambda i: d[i].mean(), J)[0]
        out['calibrated_ratio'] = float(d.mean() / dcal.mean()); out['calibrated_ratio_ci'] = boot(lambda i: d[i].mean() / dcal[i].mean(), J)[0]
        return out

    # ---------------- JEDI
    net = Jedi(64)
    RJ = net.run(xT, keep=('S', 'z', 'logits')); cJ = RJ['logits'].argmax(1)
    SF = net.run(xF, keep=('S', 'z'))
    rdJ = Ridge(SF['z'], MF['logm'], fit, dev)
    runsJ = {k: net.run(v, keep=('S', 'z', 'logits')) for k, v in E.items()}
    R['jedi'] = dict(acc_T=float((cJ == yT).mean()), acc_T_ci=wilson(int((cJ == yT).sum()), J), readout_r2_z=r2(rdJ(RJ['z']), MT['logm']),
                     W_to_Z_unedited=float((cJ[isW] == Z_).mean()))
    dJ = {k: rdJ(v['z']) - rdJ(RJ['z']) for k, v in runsJ.items()}; cJe = {k: v['logits'].argmax(1) for k, v in runsJ.items()}
    R['jedi']['edits'] = {k: stats(cJe[k], cJ, dJ[k], dJ['scale_up'], 'jedi') for k in E}
    for k in E:
        R['jedi']['edits'][k]['mcnemar_W_to_Z_vs_rotation'] = mcnemar(cJe['rotation'][isW] == Z_, cJe[k][isW] == Z_)
        R['jedi']['edits'][k]['mcnemar_calls_vs_rotation'] = mcnemar(cJe['rotation'] != cJ, cJe[k] != cJ)
        R['jedi']['edits'][k]['mcnemar_calls_vs_transl_random'] = mcnemar(cJe['transl_random'] != cJ, cJe[k] != cJ)
    log('JEDI', json.dumps(clean({k: (v['W_to_Z'], v['W_to_gq'], v['calls_changed'], round(v['calibrated_ratio'], 3)) for k, v in R['jedi']['edits'].items()})),
        'acc', R['jedi']['acc_T']); save()
    # JEDI S costs (resample ablation, same derangement scheme as X9) for the matched unit selection in the rank swap
    der = np.random.default_rng(0).permutation(J)
    while (der == np.arange(J)).any(): der = np.random.default_rng(int(der[0])).permutation(J)
    costS = np.zeros(64)
    for k in range(64):
        if RJ['S'][:, k].std() < 1e-12: continue
        Sp = RJ['S'].copy(); Sp[:, k] = RJ['S'][der, k]; m = np.zeros(64); m[k] = 1
        costS[k] = float((net.run_jetwise(xT, {'S': (m, Sp)})['logits'].argmax(1) != cJ).mean())
    topS = [int(k) for k in np.argsort(-costS)[:10]]; R['jedi']['top10_cost_S'] = topS; R['jedi']['top10_cost'] = costS[topS].tolist()
    # rank swaps for JEDI (X6 units and top-10 cost units)
    zf = xT[..., 0] / xT[..., 0].sum(1, keepdims=True); swaps = {}
    for k, j in ((13, 11), (0, 3)):
        Ja, xa = swap(xT, k); Ji, xi = swap(xT, j); com = np.intersect1d(Ja, Ji); ia, ii = np.searchsorted(Ja, com), np.searchsorted(Ji, com)
        swaps[(k, j)] = (Ja, xa, Ji, xi, com, ia, ii)
    sdS = RJ['S'].std(0); R['jedi']['rank_swap'] = []
    for (k, j), (Ja, xa, Ji, xi, com, ia, ii) in swaps.items():
        dA = net.run(xa, keep=('S',))['S'] - RJ['S'][Ja]; dI = net.run(xi, keep=('S',))['S'] - RJ['S'][Ji]
        rec = dict(edge=k, inside=j, N=int(len(com)),
                   z_gap_ratio=float(np.abs(zf[com, k] - zf[com, k + 1]).mean() / np.abs(zf[com, j] - zf[com, j + 1]).mean()))
        for lab, uu in (('x6_units', [0, 43, 51, 52, 63] if k == 13 else [1]), ('top10_cost', topS)):
            uu = [u for u in uu if sdS[u] > 0]
            ma = (np.abs(dA[:, uu]) / sdS[uu]).mean(1)[ia]; mi = (np.abs(dI[:, uu]) / sdS[uu]).mean(1)[ii]
            rec[lab] = dict(units=uu, ratio=float(ma.mean() / mi.mean()), ratio_ci=boot(lambda i: ma[i].mean() / mi[i].mean(), len(com))[0])
        R['jedi']['rank_swap'].append(rec)
    log('JEDI rank swap', json.dumps(clean(R['jedi']['rank_swap']))); save()
    # ---------------- flow models
    R['flow'] = {}
    for name in MODELS:
        t1 = time.time(); M = Flow(name)
        base = M.run(xT, keep=('L', 'h', 'logits')); cM = base['logits'].argmax(1); LT, hT = base['L'], base['h']
        LF = M.run(xF, keep=('h',)); rd = Ridge(LF['h'], MF['logm'], fit, dev)
        o = dict(acc_T=float((cM == yT).mean()), acc_T_ci=wilson(int((cM == yT).sum()), J), agreement_jedi=float((cM == cJ).mean()),
                 readout_r2_h=r2(rd(hT), MT['logm']), W_to_Z_unedited=float((cM[isW] == Z_).mean()))
        runs = {k: M.run(v, keep=('h', 'logits')) for k, v in E.items()}
        d = {k: rd(v['h']) - rd(hT) for k, v in runs.items()}; ce = {k: v['logits'].argmax(1) for k, v in runs.items()}
        o['edits'] = {k: stats(ce[k], cM, d[k], d['scale_up'], 'model') for k in E}
        for k in E:
            e = o['edits'][k]
            e['mcnemar_W_to_Z_vs_rotation'] = mcnemar(ce['rotation'][isW] == Z_, ce[k][isW] == Z_)
            e['mcnemar_calls_vs_rotation'] = mcnemar(ce['rotation'] != cM, ce[k] != cM)
            e['mcnemar_calls_vs_transl_random'] = mcnemar(ce['transl_random'] != cM, ce[k] != cM)
            wm, wj = (ce[k][isW] == Z_).astype(float), (cJe[k][isW] == Z_).astype(float)
            e['paired_W_to_Z_minus_jedi'] = float(wm.mean() - wj.mean()); e['paired_W_to_Z_minus_jedi_ci'] = boot(lambda i: wm[i].mean() - wj[i].mean(), nW)[0]
            e['paired_W_to_Z_mcnemar_vs_jedi'] = mcnemar(wj > 0, wm > 0)
        # rank swap with the model's top-10 cost latent units (as X9)
        costL = np.zeros(M.ell)
        for k in range(M.ell):
            if LT[:, k].std() < 1e-12: continue
            Lp = LT.copy(); Lp[:, k] = LT[der, k]; costL[k] = float((M.tail(Lp).argmax(1) != cM).mean())
        topu = [int(k) for k in np.argsort(-costL)[:10]]; sdL = LT.std(0); o['rank_swap'] = []
        for (k, j), (Ja, xa, Ji, xi, com, ia, ii) in swaps.items():
            dA = M.run(xa, keep=('L',))['L'] - LT[Ja]; dI = M.run(xi, keep=('L',))['L'] - LT[Ji]
            ma = (np.abs(dA[:, topu]) / sdL[topu]).mean(1)[ia]; mi = (np.abs(dI[:, topu]) / sdL[topu]).mean(1)[ii]
            o['rank_swap'].append(dict(edge=k, inside=j, N=int(len(com)), units=topu, ratio=float(ma.mean() / mi.mean()),
                                       ratio_ci=boot(lambda i: ma[i].mean() / mi[i].mean(), len(com))[0]))
        R['flow'][name] = o
        log(name, f'acc {o["acc_T"]:.4f}', json.dumps(clean({k: (round(v['W_to_Z'], 3), round(v['W_to_gq'], 3), round(v['calls_changed'], 3), round(v['calibrated_ratio'], 3)) for k, v in o['edits'].items()})),
            'swap', [(r['edge'], round(r['ratio'], 3)) for r in o['rank_swap']], f'{time.time() - t1:.0f} s'); save()
    save(); log(f'wrote {JS}; X9 verify DONE')


if __name__ == '__main__':
    main()
