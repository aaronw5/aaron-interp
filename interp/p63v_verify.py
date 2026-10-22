"""P63 verifier (loop C): fresh sample rerun of P63's key numbers plus three controls P63 did not run.

Sample: JetClass test_20M WToQQ_114, ZToQQ_114, ZJetsToNuNu_114, TTBar_114 (P63 used _112), 5,000 random rows per file
(numpy default_rng(6399), never first-k), split at random per class into fit / evaluation halves. Versions N, A, T1, T2, D are
built with P63's make_versions (same edits, part_core.build with the sign(jet_eta) fix).
Models: ParT_full; controls ParT_untr0, ParT_untr1 (P63's untrained inits).
Sites: inp, mom (model-independent), emb, b1, b3, b5, b8, cls1, cls2, pairH.
Checks
  (1) replicate: MLP partial R^2(log m | cubic log m_A) on the pool N+A+T1+T2, and on A / T rows; ridge too.
  (2) no-shift pool: the same partial fitted and scored on N+T1+T2 rows only (the rigid-shift family A, where log m - log m_A
      has s.d. 0.089 vs 0.025-0.029 elsewhere, removed).
  (3) within-jet turn test (the part of the pool where only m changes): (a) pool-fitted MLP log m read-out, slope of
      (p(T2) - p(T1)) on the true dlog m = log m(T2) - log m(T1) (c_turn) and the correlation of the two over evaluation jets;
      (b) difference probe: ridge on X(T2) - X(T1) and MLP on [X(T2) - X(T1), X(T1)] (antisymmetric augmentation), fitted on fit
      jets to predict dlog m, held-out R^2 on evaluation jets. m_A is identical in T1 and T2 (to 1e-7), so any held-out R^2 > 0 is
      true-mass information at fixed m_A, with no between-jet component.
  (4) beyond-moments partial: MLP partial R^2 of log m given [cubic log m_A + the 5 moment features of 'mom'] (reduced model also
      an MLP), i.e. what the site holds about m beyond m_A and the axis moments M0, M1, M2.
  Tests: paired Poisson bootstrap (1,000, over evaluation jets) trained - untrained for each init, p = larger one-sided p;
      Holm over the 8 network sites per check.
  (5) output: W-Z margin slope under the matched dilation (s_dil) and under the turn: rho_input = turn margin / (s_dil dlog m).
  (6) interchange at b5 and cls1 (ParT_full): 1-d ridge probe direction of r = log m - cubic(log m_A) and a 4-d DAS subspace (P63's
      das_fit); nulls: 100 random 1-d and 100 random 4-d swaps at b5 on 2,000 W/Z evaluation pairs.
  ./sync.sh && ./part_job.sh c-p63-verify 'python -m interp.p63v_verify'   -> $AI_RESULTS/part/p63_verify[_smoke].json
"""
import json, sys, time
import numpy as np
import interp.p63_true_mass_layers_part as P
from interp.p63_true_mass_layers_part import (load_rows, make_versions, load_model, Run, capture, Ridge, mlp_fit, poly3, ols, rows,
                                              r2_boot, partial_boot, slope_boot, p_greater, holm, qci, interchange, boot_pairs, das_fit,
                                              VERS, POOLV, RES_SITES, log)
from interp.part_core import cat, sub, sample, ParTNetwork, CL

SMOKE = P.SMOKE
SEED = 6399
FILES = dict(Wqq='WToQQ_114.root', Zqq='ZToQQ_114.root', QCD='ZJetsToNuNu_114.root', Tbqq='TTBar_114.root')
PER = 300 if SMOKE else 5000
NB = 100 if SMOKE else 1000
NRAND = 5 if SMOKE else 100
NNULL = 200 if SMOKE else 2000
NET_SITES = ['emb', 'b1', 'b3', 'b5', 'b8', 'cls1', 'cls2', 'pairH']
FN = P.OUT / f'p63_verify{"_smoke" if SMOKE else ""}.json'
iW, iZ = CL.index('Wqq'), CL.index('Zqq')
VI = {v: VERS.index(v) for v in VERS}


def wcorr_boot(a, b, Wb):
    def c(w):
        W = w.sum(-1, keepdims=True) if w.ndim > 1 else w.sum()
        ma = (w * a).sum(-1) / W.squeeze(); mb = (w * b).sum(-1) / W.squeeze()
        ca = (w * a * b).sum(-1) / W.squeeze() - ma * mb
        va = (w * a * a).sum(-1) / W.squeeze() - ma ** 2; vb = (w * b * b).sum(-1) / W.squeeze() - mb ** 2
        return ca / np.sqrt(np.maximum(va * vb, 1e-300))
    return float(c(np.ones(len(a)))), c(Wb)


def partial_pool(X5, O5, fj, ej, vers, Wb, dev, kind, extra5=None):
    """partial R^2 of log m given cubic(log m_A) [+ extra5 features in the reduced model] on rows `vers`"""
    Xf, jf = rows(X5, vers, fj); Xe, je = rows(X5, vers, ej)
    yf, ye = rows(O5['logm'], vers, fj)[0], rows(O5['logm'], vers, ej)[0]
    cf, ce = rows(O5['logmA'], vers, fj)[0], rows(O5['logmA'], vers, ej)[0]
    Pf, Pe = poly3(cf, cf), poly3(ce, cf)
    red = ols(Pf, yf); pr_f, pr_e = red(Pf), red(Pe)
    if extra5 is not None:
        Ef, Ee = rows(extra5, vers, fj)[0], rows(extra5, vers, ej)[0]
        a, b = mlp_fit(np.c_[Ef, Pf], yf - pr_f, jf, [np.c_[Ef, Pf], np.c_[Ee, Pe]], dev)
        pr_f, pr_e = pr_f + a, pr_e + b; Pf, Pe = np.c_[Ef, Pf], np.c_[Ee, Pe]
    rf = yf - pr_f
    if kind == 'ridge':
        m = Ridge(np.c_[Xf, Pf], rf, jf); pf_e = pr_e + m(np.c_[Xe, Pe])
    else:
        pf_e, = mlp_fit(np.c_[Xf, Pf], rf, jf, [np.c_[Xe, Pe]], dev); pf_e = pr_e + pf_e
    e, b = partial_boot(ye, pf_e, pr_e, je, Wb)
    fam = np.repeat(np.array([v[0] for v in vers]), len(ej))
    for fm in sorted(set(fam)):
        k = fam == fm; e[f'on_{fm}'] = partial_boot(ye[k], pf_e[k], pr_e[k], je[k], Wb)[0]
    return e, b


def turn_tests(X5, O5, fj, ej, Wb, dev):
    """within-jet tests on T1/T2 pairs (m_A fixed)"""
    out, boots = {}, {}
    # (a) pool-fitted MLP log m read-out
    Xf, jf = rows(X5, POOLV, fj); yf = rows(O5['logm'], POOLV, fj)[0]
    p1, p2, pN, pD = mlp_fit(Xf, yf, jf, [X5[VI['T1']][ej], X5[VI['T2']][ej], X5[VI['N']][ej], X5[VI['D']][ej]], dev)
    dt = O5['logm'][VI['T2']][ej] - O5['logm'][VI['T1']][ej]; dD = O5['logm'][VI['D']][ej] - O5['logm'][VI['N']][ej]
    jj = np.arange(len(ej))
    cT, bT = slope_boot(p2 - p1, dt, jj, Wb); cD, bD = slope_boot(pD - pN, dD, jj, Wb)
    r, br = wcorr_boot(p2 - p1, dt, Wb)
    out['pool_mlp'] = dict(c_turn=cT, c_turn_ci=qci(bT), c_dil=cD, R_turn=cT / cD, R_turn_ci=qci(bT / bD), corr_turn=r, corr_turn_ci=qci(br),
                           mean_abs_readout_change=float(np.abs(p2 - p1).mean()), mean_abs_true=float(np.abs(dt).mean()))
    boots['corr_turn'] = br
    # (b) difference probes, fitted on fit jets, antisymmetric augmentation
    D_f = X5[VI['T2']][fj] - X5[VI['T1']][fj]; dt_f = O5['logm'][VI['T2']][fj] - O5['logm'][VI['T1']][fj]
    D_e = X5[VI['T2']][ej] - X5[VI['T1']][ej]
    jf2 = np.r_[np.arange(len(fj)), np.arange(len(fj))]
    rd = Ridge(np.r_[D_f, -D_f], np.r_[dt_f, -dt_f], jf2); e, b = r2_boot(dt, rd(D_e), jj, Wb); out['diff_ridge'] = e; boots['diff_ridge'] = b
    Xa = np.r_[np.c_[D_f, X5[VI['T1']][fj]], np.c_[-D_f, X5[VI['T2']][fj]]]
    pe, = mlp_fit(Xa, np.r_[dt_f, -dt_f], jf2, [np.c_[D_e, X5[VI['T1']][ej]]], dev)
    e, b = r2_boot(dt, pe, jj, Wb); out['diff_mlp'] = e; boots['diff_mlp'] = b
    return out, boots


def main():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    P.OUT.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(SEED); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f'P63 verify start; device {dev}; smoke {SMOKE}; PER {PER}')
    Js = []
    for c, f in FILES.items():
        J, r = load_rows(P.TEST / f, PER, rng); Js.append(J); log(f'  {f}: {len(r)} random rows (first {r[:3].tolist()}), labels {np.unique(J["label"]).tolist()}')
    J0 = cat(Js); N = len(J0['label']); y = J0['label'].astype(int)
    lab = np.repeat(np.arange(len(FILES)), PER); fit = np.zeros(N, bool)
    for c in range(len(FILES)):
        ii = np.flatnonzero(lab == c); fit[rng.permutation(ii)[:len(ii) // 2]] = True
    Vs, O, keep, info = make_versions(J0, rng, dev)
    fj, ej = np.flatnonzero(fit & keep), np.flatnonzero(~fit & keep)
    R = dict(meta=dict(seed=SEED, files=FILES, per_class=PER, n_jets=N, n_kept=int(keep.sum()), n_fit=len(fj), n_eval=len(ej), NB=NB, smoke=SMOKE))
    k_ = keep
    pool = {v: dict(dlogm_q=np.quantile(O[v]['logm'][k_] - O['N']['logm'][k_], [0.05, 0.5, 0.95]).tolist(),
                    dlogmA_q=np.quantile(O[v]['logmA'][k_] - O['N']['logmA'][k_], [0.05, 0.5, 0.95]).tolist(),
                    sd_Delta=float(O[v]['Delta'][k_].std())) for v in VERS}
    pool['max_abs_dlogmA_turn'] = float(np.abs(O['T1']['logmA'][k_] - O['N']['logmA'][k_]).max())
    R['pool'] = pool; log('pool: ' + json.dumps(pool))
    # unedited accuracy on the P1 sample (ParT_full)
    JP = sample(per_class=100 if SMOKE else 1000, start=3000); pn = ParTNetwork('full'); L = pn.run(JP)['logits']
    R['accuracy_P1_sample_ParT_full'] = float((L.argmax(1) == JP['label']).mean()); del pn
    log(f"P1-sample accuracy ParT_full {R['accuracy_P1_sample_ParT_full']:.4f}")
    Wb = np.random.default_rng(SEED + 1).poisson(1.0, (NB, len(ej))).astype(np.float64)
    O5 = {k: np.stack([O[v][k] for v in VERS]) for k in ('logm', 'logmA', 'Delta')}
    cnt = (J0['mask'] > 0).sum(1)
    M1sq = lambda v: O[v]['M1e'] ** 2 + O[v]['M1p'] ** 2
    MOM = np.stack([np.stack([np.log(O[v]['M0']), O[v]['M1e'] / O[v]['M0'], O[v]['M1p'] / O[v]['M0'], np.log(O[v]['M2'] / O[v]['M0']),
                              M1sq(v) / (O[v]['M0'] * O[v]['M2'])], 1) for v in VERS]).astype(np.float32)
    RES, BO, LOG = {}, {}, {}
    keep_runs = {}
    for nm in ('ParT_full', 'ParT_untr0', 'ParT_untr1'):
        t1 = time.time(); pn = load_model(nm); run = Run(pn)
        Fe, Lg = capture(run, Vs, 'full', cnt)
        Fe = {s: Fe[s] for s in (['inp'] if nm == 'ParT_full' else []) + NET_SITES}
        if nm == 'ParT_full': Fe['mom'] = MOM
        acc = float((Lg[0][keep].argmax(1) == y[keep]).mean())
        mg = lambda L: L[:, iW] - L[:, iZ]
        wzE = np.zeros(N, bool); wzE[ej] = True; wzE &= (y == iW) | (y == iZ)
        dmD = mg(Lg[VI['D']]) - mg(Lg[VI['N']]); dlD = O['D']['logm'] - O['N']['logm']
        s_dil = float((dmD[wzE] * dlD[wzE]).sum() / (dlD[wzE] ** 2).sum())
        dmT = mg(Lg[VI['T2']]) - mg(Lg[VI['T1']]); dlT = O['T2']['logm'] - O['T1']['logm']
        jjw = np.flatnonzero(wzE[ej]); ejw = ej[wzE[ej]]; a_ = dmT[ejw] * s_dil * dlT[ejw]; c_ = (s_dil * dlT[ejw]) ** 2
        wW = keep & (y == iW)
        LOG[nm] = dict(acc_this_sample=acc, s_dil=s_dil, rho_input=float(a_.sum() / c_.sum()), rho_input_ci=qci((Wb[:, jjw] @ a_) / (Wb[:, jjw] @ c_)),
                       W_called_Z={v: float((Lg[VI[v]][wW].argmax(1) == iZ).mean()) for v in VERS},
                       W_called_Z_A006=float((Lg[VI['A']][wW & (info['shift_dose'] == 0.06)].argmax(1) == iZ).mean()),
                       W_called_Z_N_at_A006=float((Lg[VI['N']][wW & (info['shift_dose'] == 0.06)].argmax(1) == iZ).mean()))
        log(f'{nm}: {json.dumps(LOG[nm])} ({time.time() - t1:.0f} s)')
        RES[nm], BO[nm] = {}, {}
        for s in Fe:
            t2 = time.time(); X5 = Fe[s]; r_, b_ = {}, {}
            for kind in ('ridge', 'mlp'):
                r_[f'pool_{kind}'], b_[f'pool_{kind}'] = partial_pool(X5, O5, fj, ej, POOLV, Wb, dev, kind)
            r_['noA_mlp'], b_['noA_mlp'] = partial_pool(X5, O5, fj, ej, ['N', 'T1', 'T2'], Wb, dev, 'mlp')
            if s != 'mom': r_['beyond_mom_mlp'], b_['beyond_mom_mlp'] = partial_pool(X5, O5, fj, ej, POOLV, Wb, dev, 'mlp', extra5=MOM)
            tt, tb = turn_tests(X5, O5, fj, ej, Wb, dev); r_['turn'] = tt; b_.update({f'turn_{k}': v for k, v in tb.items()})
            RES[nm][s], BO[nm][s] = r_, b_
            log(f'  {nm} {s:6s} ({time.time() - t2:.0f} s): partial pool ridge {r_["pool_ridge"]["r2"]:.3f} mlp {r_["pool_mlp"]["r2"]:.3f} '
                f'(A {r_["pool_mlp"]["on_A"]["r2"]:.3f}, T {r_["pool_mlp"]["on_T"]["r2"]:.3f}) | noA mlp {r_["noA_mlp"]["r2"]:.3f} '
                f'| beyond-mom {r_.get("beyond_mom_mlp", {}).get("r2")} | turn: R_turn {tt["pool_mlp"]["R_turn"]:.3f} corr {tt["pool_mlp"]["corr_turn"]:.3f} '
                f'diff-ridge R2 {tt["diff_ridge"]["r2"]:.3f} diff-mlp R2 {tt["diff_mlp"]["r2"]:.3f}')
        if nm == 'ParT_full': keep_runs['full'] = (run, Lg, Fe)
        else: del run, pn
        torch.cuda.empty_cache() if dev == 'cuda' else None
    R['logits'] = LOG; R['readouts'] = RES
    # tests trained vs untrained, Holm over network sites
    tests = {}
    for key in ('pool_mlp', 'pool_ridge', 'noA_mlp', 'beyond_mom_mlp', 'turn_diff_mlp', 'turn_diff_ridge', 'turn_corr_turn'):
        rows_, ps = [], []
        for s in NET_SITES:
            bT = BO['ParT_full'][s][key]; pv = [p_greater(bT - BO[u][s][key]) for u in ('ParT_untr0', 'ParT_untr1')]
            get = lambda nm: (RES[nm][s]['turn'][key[5:]]['r2'] if key in ('turn_diff_mlp', 'turn_diff_ridge') else
                              RES[nm][s]['turn']['pool_mlp']['corr_turn'] if key == 'turn_corr_turn' else RES[nm][s][key]['r2'])
            rows_.append(dict(site=s, trained=get('ParT_full'), untr0=get('ParT_untr0'), untr1=get('ParT_untr1'),
                              diff_ci_untr0=qci(bT - BO['ParT_untr0'][s][key]), diff_ci_untr1=qci(bT - BO['ParT_untr1'][s][key]), p=max(pv)))
            ps.append(max(pv))
        for r_, ph in zip(rows_, holm(ps)): r_['p_holm'] = ph
        tests[key] = rows_
        log(f'test {key}: ' + ', '.join(f'{r_["site"]} {r_["trained"]:.3f}/{r_["untr0"]:.3f}/{r_["untr1"]:.3f} p_holm {r_["p_holm"]:.3g}' for r_ in rows_))
    R['tests_vs_untrained'] = tests
    # interchange (ParT_full)
    run, Lg, Fe = keep_runs['full']; s_dil = LOG['ParT_full']['s_dil']
    probes = {}
    for s in ('b5', 'cls1', 'cls2'):
        Xf, jf = rows(Fe[s], POOLV, fj); yf = rows(O5['logm'], POOLV, fj)[0]; cf = rows(O5['logmA'], POOLV, fj)[0]
        red = ols(poly3(cf, cf), yf); probes[s] = Ridge(Xf, yf - red(poly3(cf, cf)), jf)
    pc2 = lambda C: probes['cls2'](C)
    ICH = {}
    for site in ('b5', 'cls1'):
        t1 = time.time(); g = probes[site].grad_raw(); u = g / max(np.linalg.norm(g), 1e-30)
        Qs = {'probe1d': torch.from_numpy(u[:, None].astype(np.float32)).to(run.dev)}
        Q4, l0, l1 = das_fit(run, Vs, 'full', cnt, fj, site, 4, rng); Qs['das4'] = Q4
        r_ = interchange(run, Vs, O, 'full', cnt, y, ej, site, Qs, pc2, rng, s_dil)
        for q in r_: r_[q].update(boot_pairs(r_[q].pop('_per'), Wb))
        r_['das4']['loss_first10'], r_['das4']['loss_last20'] = l0, l1
        ent = dict(main=r_)
        if site == 'b5':
            ejw = ej[(y[ej] == iW) | (y[ej] == iZ)]; sel = np.sort(rng.choice(ejw, min(NNULL, len(ejw)), replace=False)); Qn = {}
            for i in range(NRAND):
                v_ = rng.standard_normal(128); Qn[f'r1_{i}'] = torch.from_numpy((v_ / np.linalg.norm(v_))[:, None].astype(np.float32)).to(run.dev)
                Qn[f'r4_{i}'] = torch.linalg.qr(torch.from_numpy(rng.standard_normal((128, 4)).astype(np.float32)))[0].to(run.dev)
            Qn['probe1d'], Qn['das4'] = Qs['probe1d'], Qs['das4']
            rn = interchange(run, Vs, O, 'full', cnt, y, sel, site, Qn, pc2, rng, s_dil, null=True); nul = {}
            for dim, ref in (('r1', 'probe1d'), ('r4', 'das4')):
                xs = [rn[q] for q in rn if q.startswith(dim + '_')]
                for met in ('iia_informative', 'rho', 'calls_changed'):
                    vals = np.array([z[met] for z in xs if z[met] is not None], float); ov = rn[ref][met]
                    nul[f'{ref}_{met}'] = dict(obs=ov, null_q=np.quantile(vals, [0.025, 0.5, 0.975]).tolist() if len(vals) else None,
                                               p_rank=float((1 + (np.abs(vals) >= abs(ov)).sum()) / (len(vals) + 1)) if (len(vals) and ov is not None) else None)
            ent['null'] = nul
        ICH[site] = ent
        log(f'interchange @ {site} ({time.time() - t1:.0f} s): ' + ' | '.join(
            f'{q}: rho {r_[q]["rho"]:.4f} {np.round(r_[q]["rho_ci"], 4).tolist()} IIA-inf {r_[q]["iia_informative"]} calls {r_[q]["calls_changed"]:.4f} '
            f'cls2-readout {r_[q]["readout_cls2_shift"]:.4f} McNemar {r_[q]["mcnemar_correct"]}' for q in ('probe1d', 'das4')))
        if 'null' in ent: log('  null: ' + json.dumps(ent['null']))
    R['interchange'] = ICH
    FN.write_text(json.dumps(R, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else (float(o) if isinstance(o, np.generic) else str(o))))
    log(f'wrote {FN}'); log('done')


if __name__ == '__main__':
    main()
