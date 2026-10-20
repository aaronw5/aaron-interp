"""X10 verification (skeptical rerun of interp.x10_das_jedi on fresh samples, plus controls the X10 run did not have).

  ./job.sh x10-verify 'python -m interp.x10_verify'            (1 CPU / 2 GiB)
  ./job.sh x10-verify-smoke 'python -m interp.x10_verify --smoke'
  ./job.sh x10-verify2 'python -m interp.x10_verify --paired'   (same run + paired bootstrap of V-DAS minus class-transfer DAS and minus
                                                                 full swap; no calibration; writes x10_verify_paired.json)

Same code path as X10 (JNet exact fixed-point tail, das_fit with straight-through gradient, range clip Pi, scores, boot_eff,
random orthonormal nulls). Differences:
 Samples. Fit F' = sample(per_class=3000, seed=23); test T' = sample(per_class=3000, seed=29); every jet of X10's F (seed 0) and
   T (seed 17, per_class 4000) and of F' is removed from T', and X10's F/T jets are removed from F'. Train pairs 6,000 from F'
   (seed 32); test pairs 4,000 from T' (seed 31), random arm only.
 Per stage (S, h4, h5), V (M0, M2, D, WZ), k (1, 2, 4): DAS (same hyper-parameters and init seeds as X10), effective IIA with
   1000-resample bootstrap CI, 1000 random subspaces (p_rot, Holm over 36), transfer to the other V.
 Added controls / diagnostics:
   (a) source-call baseline: fraction of E_H pairs whose counterfactual call equals the source jet's own call (= effective IIA
       of the full swap Q = I); a "class-transfer" DAS trained on label = source call, scored on every V's counterfactuals.
   (b) range-clip dependence: the same DAS Q evaluated without the clip (exact tail, wrap); share of E_H pairs in which the clip
       changes at least one live unit (unit with a non-empty quantizer range); effective IIA on clipped vs unclipped pairs;
       norm of the clip correction relative to the swapped component; same for 50 random Q.
   (c) representation check (V with an input edit): in the learned subspace, the share of the base-to-source gap closed by the
       input edit, 1 - sum|Q^T(h_edit - h_s)|^2 / sum|Q^T(h_b - h_s)|^2, and the share of the edit's displacement at the stage that
       lies inside span(Q), vs 200 random Q.
   (d) calibration at S and h4 (X10's calibrate(), unchanged) on T'.
Writes $AI_RESULTS/jedi/x10_verify.json (x10_verify_smoke.json with --smoke).
"""
import argparse, json, time
import numpy as np
from interp.x10_das_jedi import (OUT, VARS, EDITV, KS, STAGES, clean, holm, moments, do_edit, wz_call, indices, JNet, das_fit,
                                 proj, scores, boot_eff, boot_diff, rand_Q, calibrate)

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def main(a):
    sm = a.smoke; JS = OUT / ('x10_verify_smoke.json' if sm else ('x10_verify_paired.json' if a.paired else 'x10_verify.json')); OUT.mkdir(parents=True, exist_ok=True)
    R = dict(smoke=sm)
    save = lambda: JS.write_text(json.dumps(clean(R), indent=1))
    from interp.jedi_core import sample
    pc = 300 if sm else 3000
    xF, yF, _ = sample(pc, 23); xT, yT, _ = sample(pc, 29)
    iF, iT = indices(pc, 23), indices(pc, 29); old = np.union1d(indices(4000, 0), indices(4000, 17))
    kF = ~np.isin(iF, old); kT = ~np.isin(iT, np.union1d(old, iF))
    xF, yF, xT, yT = xF[kF], yF[kF], xT[kT], yT[kT]; NF, NT = len(yF), len(yT)
    R['N'] = dict(F=NF, T=NT, F_removed=int((~kF).sum()), T_removed=int((~kT).sum())); log('N', R['N'])
    mF, mT = moments(xF), moments(xT)
    npair = 800 if sm else 6000; ntest = 600 if sm else 4000
    r2 = np.random.default_rng(32); fb = r2.integers(0, NF, npair); fs = r2.integers(0, NF, npair)
    while (fb == fs).any(): fs[fb == fs] = r2.integers(0, NF, (fb == fs).sum())
    r1 = np.random.default_rng(31); tb = r1.integers(0, NT, ntest); ts = r1.integers(0, NT, ntest)
    while (tb == ts).any(): ts[tb == ts] = r1.integers(0, NT, (tb == ts).sum())
    R['test_pairs_sharing_base'] = float(1 - len(np.unique(tb)) / ntest)

    net = JNet(); RF = net.reps(xF); RT = net.reps(xT)
    cF = RF['logits'].argmax(1); cT = RT['logits'].argmax(1)
    R['acc_T'] = float((cT == yT).mean()); R['acc_F'] = float((cF == yF).mean()); log('acc', R['acc_T'], R['acc_F'])
    R['tail_check'] = net.check(xT[:2000], {k: v[:2000] for k, v in RT.items()}); save()

    lab = dict(train=dict(base=cF[fb], src=cF[fs]), test=dict(base=cT[tb], src=cT[ts])); RE = {}
    for V in EDITV:
        lab['train'][V] = net.calls(do_edit(xF[fb], V, (mF[0][fb], mF[1][fb]), (mF[0][fs], mF[1][fs])))
        RE[V] = net.reps(do_edit(xT[tb], V, (mT[0][tb], mT[1][tb]), (mT[0][ts], mT[1][ts])), keep=('S', 'h4', 'h5', 'logits'))
        lab['test'][V] = RE[V]['logits'].argmax(1)
    lab['train']['WZ'] = wz_call(RF['logits'][fb], RF['logits'][fs]); lab['test']['WZ'] = wz_call(RT['logits'][tb], RT['logits'][ts])
    bT, sT = lab['test']['base'], lab['test']['src']
    R['cf_change_frac'] = {V: float((lab['test'][V] != bT).mean()) for V in VARS}
    # (a) source-call baseline: on E_H, how often is the counterfactual call the source's own call
    R['source_call_baseline'] = {V: dict(eff_full_swap=float((lab['test'][V] == sT)[lab['test'][V] != bT].mean()), n_eff=int((lab['test'][V] != bT).sum()))
                                 for V in VARS}
    log('cf change', R['cf_change_frac']); log('source-call baseline', R['source_call_baseline']); save()

    nrand = 50 if sm else 1000; steps = 100 if sm else 800; B = 100 if sm else 1000
    R['stages'] = {}; Qkeep = {}
    for st in STAGES:
        ft = net.tail(st); d = RT[st].shape[1]; rg = net.rng_[st]; live = rg[1] > rg[0]
        hb, hs = RT[st][tb], RT[st][ts]; res = dict(d=d, live_units=int(live.sum()), k={})

        def clipinfo(Q, EH):
            sw = ((hs - hb) @ Q) @ Q.T; hraw = hb + sw; hc = proj(rg, hb, hraw); cor = (hc - hraw)[:, live]
            cl = (np.abs(cor) > 1e-7).any(1)
            rel = np.linalg.norm(cor, axis=1) / np.maximum(np.linalg.norm(sw[:, live], axis=1), 1e-12)
            return hraw, cl, dict(frac_EH_pairs_clipped=float(cl[EH].mean()), median_rel_clip_norm_EH=float(np.median(rel[EH])))

        def repcheck(Q, V):
            he = RE[V][st]; gap = (((he - hs) @ Q) ** 2).sum() / max((((hb - hs) @ Q) ** 2).sum(), 1e-12)
            dv = he - hb; inside = (((dv @ Q) ** 2).sum()) / max((dv ** 2).sum(), 1e-12)
            return float(1 - gap), float(inside)

        for k in KS:
            rk = dict(das={}, cls={}, random={})
            rr = np.random.default_rng(1000 + k); null = {V: [] for V in VARS}; nclip = {V: [] for V in VARS}; nrep = {V: [] for V in EDITV}
            for j in range(nrand):
                Q = rand_Q(d, k, rr); c = net.exact(st, proj(rg, hb, hb + ((hs - hb) @ Q) @ Q.T))[0].argmax(1)
                for V in VARS:
                    s_ = scores(c, lab['test'][V], bT); null[V].append(s_['eff'] if s_['eff'] is not None else np.nan)
                    if j < 50: nclip[V].append(clipinfo(Q, lab['test'][V] != bT)[2])
                    if j < 200 and V in EDITV: nrep[V].append(repcheck(Q, V))
            # (a) class-transfer DAS: label = source's own call
            Qc, infc = das_fit(ft, RF[st][fb], RF[st][fs], lab['train']['src'], lab['train']['base'], k, seed=40 + k, steps=steps, rg=rg)
            cc = net.exact(st, proj(rg, hb, hb + ((hs - hb) @ Qc) @ Qc.T))[0].argmax(1)
            rk['cls'] = dict(train=infc, eff_own=scores(cc, sT, bT)['eff'], eff_on={V: scores(cc, lab['test'][V], bT)['eff'] for V in VARS})
            for V in VARS:
                Q, info = das_fit(ft, RF[st][fb], RF[st][fs], lab['train'][V], lab['train']['base'], k, seed=10 + k, steps=steps, rg=rg)
                LV = lab['test'][V]; EH = LV != bT
                hraw, cl, ci = clipinfo(Q, EH)
                c = net.exact(st, proj(rg, hb, hraw))[0].argmax(1); s_ = scores(c, LV, bT)
                s_['eff_ci'] = boot_eff(c, LV, bT, B=B)
                nv = np.asarray(null[V]); fin = np.isfinite(nv)
                s_['p_rot'] = float((1 + (nv[fin] >= s_['eff']).sum()) / (1 + fin.sum())) if s_['eff'] is not None and fin.any() else None
                s_['transfer'] = {V2: scores(c, lab['test'][V2], bT)['eff'] for V2 in VARS}
                cnc = net.exact(st, hraw)[0].argmax(1); ok = c == LV
                s_['eff_noclip'] = scores(cnc, LV, bT)['eff']
                s_['clip'] = dict(ci, eff_unclipped_pairs=float(ok[EH & ~cl].mean()) if (EH & ~cl).sum() >= 20 else None, n_unclipped=int((EH & ~cl).sum()),
                                  eff_clipped_pairs=float(ok[EH & cl].mean()) if (EH & cl).sum() >= 20 else None, n_clipped=int((EH & cl).sum()))
                s_['train'] = info
                if a.paired:  # paired bootstrap: V-DAS minus class-transfer DAS / minus full swap (= source call), on E_H of V
                    s_['minus_cls'] = boot_diff(c, cc, LV, bT, B=B); s_['minus_full_swap'] = boot_diff(c, sT, LV, bT, B=B)
                if V in EDITV:
                    g, ins = repcheck(Q, V); nr = np.asarray(nrep[V])
                    s_['rep'] = dict(gap_closed=g, edit_disp_inside=ins, random_gap_closed_q=np.quantile(nr[:, 0], [.025, .5, .975]).tolist(),
                                     random_inside_q=np.quantile(nr[:, 1], [.025, .5, .975]).tolist())
                    if k in (1, 2): Qkeep[(st, k, V)] = Q
                rk['das'][V] = s_
                rk['random'][V] = dict(mean=float(np.nanmean(nv)), q=np.nanquantile(nv, [.025, .5, .975]).tolist() if fin.any() else None,
                                       clip_frac_EH_median=float(np.median([z['frac_EH_pairs_clipped'] for z in nclip[V]])),
                                       clip_rel_norm_median=float(np.median([z['median_rel_clip_norm_EH'] for z in nclip[V]])))
                log(f'{st} k={k} {V}: eff {s_["eff"]} {s_["eff_ci"]} n {s_["n_eff"]} p_rot {s_["p_rot"]}; random {rk["random"][V]["mean"]:.3f}; '
                    f'noclip {s_["eff_noclip"]}; clip {s_["clip"]}; rep {s_.get("rep")}; transfer {s_["transfer"]}')
            log(f'{st} k={k} class-transfer DAS: {rk["cls"]}')
            res['k'][k] = rk; R['stages'][st] = res; save()
    rows = [(st, k, V, R['stages'][st]['k'][k]['das'][V]['p_rot']) for st in STAGES for k in KS for V in VARS]
    adj = holm([r[3] if r[3] is not None else 1.0 for r in rows])
    R['holm'] = dict(n_tests=len(rows), n_below_005=int((adj < 0.05).sum()), adj=[dict(stage=r[0], k=r[1], V=r[2], p=r[3], holm=float(h)) for r, h in zip(rows, adj)])
    log('Holm', R['holm']['n_below_005'], '/', len(rows)); save()
    # (d) calibration at S and h4 (X10's function, unchanged)
    if a.paired: log(f'wrote {JS}; paired DONE'); return
    R['calibration'] = {}
    for st in ('S', 'h4'):
        cres = dict(k={k: dict(das={V: {'_Q': Qkeep[(st, k, V)]} for V in EDITV}) for k in (1, 2)})
        R['calibration'][st] = calibrate(net, st, cres, xF, xT, yT, RF, sm); save()
    log(f'wrote {JS}; X10 verify DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--paired', action='store_true'); main(ap.parse_args())
