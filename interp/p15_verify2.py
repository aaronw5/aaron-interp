"""P15 verification, round 2 (loop C): a third sample plus controls that round 1 (interp/p15_verify.py) did not run.

Sample: JetClass val_5M <sample>_124.root (P15 used _120, round 1 _122), 2,000 random rows per class (numpy default_rng(45)),
fit/test halves default_rng(46); same build and edits as interp/p15_particlenet_internal.py (functions imported, not copied).
Parts:
  (a) Sign test. Edits phi -0.03/-0.06/-0.09 and eta -0.06 (invariant mass fixed; m_A rises for either sign). The model's response
      to each sign; the P15 DAS direction (fitted on phi +0.03/+0.06/+0.09 only) evaluated on the opposite-sign counterfactual:
      share of the change in d = l_W - l_Z, IIA, per-jet r, and the mean change of the direction's jet projection.
      Competing predictions: a direction that encodes |axis offset| or m_A moves the same way under +/- shifts and reproduces the
      -0.06 response (share near the +0.06 value); a direction that encodes the signed mean offset (a translation coordinate) moves
      the opposite way and gives a negative or small share for -0.06 when the model's response has the same sign for both shifts.
  (b) One direction fitted on both signs (phi +/-0.03/0.06/0.09): share for +0.06 and -0.06 separately.
  (c) Random directions (5) and the P15-style per-site DAS on this sample (third replicate of the per-site shares; ec3 differed
      between P15 (0.11) and round 1 (0.57)).
  (d) Graph freezes for phi +0.06 and -0.06.
  (e) Atlas control: pooled channels vs random rotations of (i) the centred channels (as P15) and (ii) the standardised live channels;
      and the dependence of a channel's max |Spearman rho| on its fraction of exact zeros (ReLU sparsity), dense-channel subset.
  (f) Pooled-channel literal circuits (K = 8..64, 100 random sets), third replicate.
  ./sync.sh && ./part_job.sh c-p15-verify2 'python -m interp.p15_verify2'   -> $AI_RESULTS/part/p15_verify2.json
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import interp.p15_particlenet_internal as P

P.SEED = 45
P.LEVELS = ['pool']
P.GRAPH_EDITS = ['phi0.06', 'phim0.06']
FILE_ID = 124
OUTV = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
FN = OUTV / 'p15_verify2.json'
T0 = time.time()
POS_CF = ['phi0.03', 'phi0.06', 'phi0.09']
NEG_CF = ['phim0.03', 'phim0.06', 'phim0.09']
EV = ['phi0.06', 'phim0.06', 'eta0.06', 'etam0.06', 'angk', 'pt1.25']


def log(*a):
    print(f'[{time.time() - T0:6.0f} s]', *a, flush=True)


def load_class(c, rng):
    import uproot
    pc = P.pc; path = P.VAL / f'{pc.PRE[c]}_{FILE_ID}.root'
    N = uproot.open(path)['tree'].num_entries
    rows = np.sort(rng.choice(N, P.PER, replace=False)); parts = []
    for s in range(0, N, 25000):
        r = rows[(rows >= s) & (rows < s + 25000)]
        if len(r): parts.append(pc.sub(pc.read_root(path, s, min(s + 25000, N)), r - s))
    J = pc.cat(parts); assert (J['label'].astype(int) == pc.CL.index(c)).all(), c
    return J, rows, N


def stats(model, XWb, site, q, MW, LW, testW, Wt, cfs):
    ev = P.das_eval(model, XWb, site, q, MW, LW, testW, cfs); st, B = P.das_stats(ev, Wt); o = {}
    qn = q.double().cpu().numpy()
    for e in cfs:
        r, _ = P.wcorr(Wt, ev[e]['dd_int'], ev[e]['dd_cf'])
        ds = (MW[e][site][testW] - MW['base'][site][testW]) @ qn
        o[e] = dict(share=st[e]['share_dd'], iia=st[e]['iia'], r=r, mean_dd_cf=st[e]['mean_dd_cf'], mean_ds=float(ds.mean()),
                    mean_ds_ci=P.ci((Wt @ ds) / Wt.sum(1)))
    return o, B


def atlas_ctrl(Y, X, y, fit, test, rng):
    """P15 atlas comparison with two rotation controls and the sparsity dependence"""
    from scipy.stats import spearmanr, mannwhitneyu
    live = Y.std(0) > 1e-6; Yl = Y[:, live]; fz = (Yl <= 1e-9).mean(0)
    A0 = P.atlas(Yl, X, y, fit, test, rng)                                   # rotation of the centred channels (P15)
    Ys = (Yl - Yl.mean(0)) / Yl.std(0)
    A1 = P.atlas(Ys, X, y, fit, test, np.random.default_rng(int(rng.integers(1 << 30))))   # rotation of the standardised channels
    mx = np.array([max(abs(t[1]) for t in c['top']) for c in A0['channels']]); cond = np.array([c['cond_r2'] for c in A0['channels']])
    out = dict(n_live=int(live.sum()), frac_zero_quantiles=np.percentile(fz, [10, 25, 50, 75, 90]).tolist(),
               rotation_centred={k: dict(channels=v['channels_median'], rotated=v['rotated_median'], p=v['mannwhitney_p']) for k, v in A0['comparison'].items()},
               rotation_standardised={k: dict(channels=v['channels_median'], rotated=v['rotated_median'], p=v['mannwhitney_p']) for k, v in A1['comparison'].items()},
               spearman_maxrho_vs_fraczero=float(spearmanr(mx, fz)[0]), spearman_condr2_vs_fraczero=float(spearmanr(cond, fz)[0]))
    # rotated directions' max |rho| (standardised rotation), for the dense-channel comparison
    Q, _ = np.linalg.qr(np.random.default_rng(7).standard_normal((Ys.shape[1], Ys.shape[1]))); Yr = Ys @ Q
    from scipy.stats import rankdata
    RX = rankdata(X, axis=0); RX = (RX - RX.mean(0)) / (RX.std(0) + 1e-12)
    RY = rankdata(Yr, axis=0); RY = (RY - RY.mean(0)) / (RY.std(0) + 1e-12); mxr = np.abs(RY.T @ RX / len(X)).max(1)
    for thr in (0.2, 0.4):
        d = fz < thr
        out[f'dense_lt{thr}'] = dict(n=int(d.sum()), channels_median=float(np.median(mx[d])) if d.any() else None, rotated_median=float(np.median(mxr)),
                                     p=float(mannwhitneyu(mx[d], mxr).pvalue) if d.sum() > 2 else None)
    return out


def main():
    P._imports(); torch = P.torch; pc = P.pc; p4 = P.p4; pn_net = P.pn_net; FEATURES = P.FEATURES
    P.DEV = 'cuda' if torch.cuda.is_available() else 'cpu'; OUTV.mkdir(parents=True, exist_ok=True)
    PNDIR = Path(os.environ.get('JETDISTILL_MODELS', '/j-jepa-vol/aaron-interp/models/part')).parent / 'particlenet'
    res = dict(file_id=FILE_ID, seed=45, per_class=P.PER, n_boot=P.NB, steps=P.STEPS, device=P.DEV, models={})
    rng = np.random.default_rng(45); parts = []
    for c in pc.CL:
        J, r, N = load_class(c, rng); parts.append(J); log(f'loaded {c}: {len(r)} of {N}')
    J0 = pc.cat(parts); del parts; y = J0['label'].astype(int); N = len(y)
    pt, de, dp, ms = pc.kin(J0); real = (pt > 0).astype(np.float32); Jb = pc.build(J0, pt, de, dp, ms)
    perm = np.random.default_rng(46).permutation(N); is_test = np.zeros(N, bool); is_test[perm[N // 2:]] = True
    iW, iZ = P.iW, P.iZ
    WZ = np.where((y == iW) | (y == iZ))[0]; fitW = np.where(~is_test[WZ])[0]; testW = np.where(is_test[WZ])[0]
    fit_all = np.where(~is_test)[0]; test_all = np.where(is_test)[0]
    res['sample'] = dict(N=N, n_wz=len(WZ), n_wz_fit=len(fitW), n_wz_test=len(testW))
    JW0 = pc.sub(J0, WZ); yW = y[WZ]; ptW, deW, dpW, msW = pt[WZ], de[WZ], dp[WZ], ms[WZ]; rW = real[WZ]
    JEW = {'base': pc.sub(Jb, WZ)}; OBW = {'base': P.basic_obs(JEW['base'])}
    for dd_ in (0.03, 0.06, 0.09):
        JEW[f'phi{dd_}'] = pc.build(JW0, ptW, deW, dpW + dd_ * rW, msW); JEW[f'phim{dd_}'] = pc.build(JW0, ptW, deW, dpW - dd_ * rW, msW)
    JEW['eta0.06'] = pc.build(JW0, ptW, deW + 0.06 * rW, dpW, msW); JEW['etam0.06'] = pc.build(JW0, ptW, deW - 0.06 * rW, dpW, msW)
    OBW['phi0.06'] = P.basic_obs(JEW['phi0.06']); kmatch = float(np.exp((OBW['phi0.06']['log_mA'] - OBW['base']['log_mA']).mean()))
    JEW['angk'] = pc.build(JW0, ptW, deW * kmatch, dpW * kmatch, msW); JEW['ang1.1'] = pc.build(JW0, ptW, deW * 1.1, dpW * 1.1, msW)
    JEW['pt1.25'] = pc.build(JW0, ptW * 1.25, deW, dpW, msW * 1.25)
    a = 0.3; JEW['rot0.3'] = pc.build(JW0, ptW, (deW * np.cos(a) - dpW * np.sin(a)) * rW, (deW * np.sin(a) + dpW * np.cos(a)) * rW, msW)
    ALL_E = list(P.EDITS) + NEG_CF + ['etam0.06']
    for e in ALL_E: OBW[e] = P.basic_obs(JEW[e]); assert np.array_equal(JEW[e]['mask'], JEW['base']['mask']), e
    res['edits'] = dict(k_match=kmatch, checks={e: dict(dlogm=float((OBW[e]['log_m'] - OBW['base']['log_m']).mean()),
                                                        dlogm_absmax=float(np.abs(OBW[e]['log_m'] - OBW['base']['log_m']).max()),
                                                        dlogmA=float((OBW[e]['log_mA'] - OBW['base']['log_mA']).mean()),
                                                        frac_dlogmA_pos=float(((OBW[e]['log_mA'] - OBW['base']['log_mA']) > 0).mean())) for e in ALL_E})
    log(f'sample N {N}, W/Z {len(WZ)} (fit {len(fitW)}, test {len(testW)}), k_match {kmatch:.4f}')
    for e, v in res['edits']['checks'].items(): log(f'   edit {e}: {v}')
    Eall = P.efps(Jb); OBS_ALL = P.atlas_obs(Jb, Eall); log('atlas observables computed')
    Wt = P.bootW(len(testW), 48); dfun = lambda L: L[:, iW] - L[:, iZ]
    for n in ('full', 'kin'):
        t1 = time.time(); key = f'ParticleNet_{n}'; nf = len(FEATURES[n]); R = {}
        model = P.freeze(pn_net.load_particlenet(PNDIR / f'ParticleNet_{n}.pt', nf, P.DEV))
        Xall = P.to_dev(p4.pn_inputs(Jb, n)); RA = P.fset(model, Xall, np.arange(N)); Lall = RA['L']
        R['acc'] = float((Lall.argmax(1) == y).mean())
        XW = {e: P.to_dev(p4.pn_inputs(JEW[e], n)) for e in ['base'] + ALL_E}
        RW = {e: P.fset(model, XW[e], np.arange(len(WZ))) for e in ['base'] + ALL_E}
        LW = {e: RW[e]['L'] for e in RW}; MW = {e: {s: P.site_val(RW[e], s) for s in P.SITES} for e in RW}
        isW_t = yW[testW] == iW
        R['W_to_Z_base'] = float((LW['base'].argmax(1)[yW == iW] == iZ).mean())
        R['edit_response'] = {}
        for e in ALL_E:
            dd = (dfun(LW[e]) - dfun(LW['base']))[testW]
            R['edit_response'][e] = dict(W_to_Z=float((LW[e].argmax(1)[yW == iW] == iZ).mean()), mean_dd_test=float(dd.mean()),
                                         mean_dd_test_ci=P.ci((Wt @ dd) / Wt.sum(1)), mean_dd_test_W=float(dd[isW_t].mean()), mean_dd_test_Z=float(dd[~isW_t].mean()))
        # paired: per-jet correlation of the +0.06 and -0.06 responses
        ddp = (dfun(LW['phi0.06']) - dfun(LW['base']))[testW]; ddm = (dfun(LW['phim0.06']) - dfun(LW['base']))[testW]
        R['corr_dd_phi_plus_minus'] = P.wcorr(Wt, ddp, ddm)[0]
        log(f'{key}: acc {100 * R["acc"]:.2f} %; W->Z base {R["W_to_Z_base"]:.3f}; ' + '; '.join(f'{e}: W->Z {v["W_to_Z"]:.3f} dd {v["mean_dd_test"]:+.3f}' for e, v in R['edit_response'].items())
            + f'; r(dd+, dd-) {R["corr_dd_phi_plus_minus"]["est"]:.3f}')
        rng_d = np.random.default_rng(55); R['das'] = {}; tests = []
        for site in P.SITES:
            t2 = time.time(); D = {}
            P.DAS_CF = POS_CF; q, _ = P.das_train(model, XW['base'], site, MW, LW, fitW, rng_d)
            D['pos_fit'], Bp = stats(model, XW['base'], site, q, MW, LW, testW, Wt, EV)
            diff = Bp['phim0.06']['share'] - Bp['phi0.06']['share']
            D['pos_fit']['share_minus_vs_plus'] = dict(est=D['pos_fit']['phim0.06']['share']['est'] - D['pos_fit']['phi0.06']['share']['est'], ci=P.ci(diff), **P.p_two(diff))
            D['pos_fit']['p_share_minus_eq0'] = P.p_two(Bp['phim0.06']['share'])
            tests.append((site, D['pos_fit']['share_minus_vs_plus']['p']))
            P.DAS_CF = POS_CF + NEG_CF; q2, _ = P.das_train(model, XW['base'], site, MW, LW, fitW, rng_d)
            D['both_fit'], _ = stats(model, XW['base'], site, q2, MW, LW, testW, Wt, ['phi0.06', 'phim0.06', 'eta0.06', 'etam0.06'])
            D['both_fit']['abs_cos_with_pos_fit'] = float(abs((q @ q2).item()))
            rs = []
            for r in range(P.NRAND):
                qr = torch.randn(P.DIM[site], generator=torch.Generator().manual_seed(4500 + r)).to(P.DEV); qr = qr / qr.norm()
                o, _ = stats(model, XW['base'], site, qr, MW, LW, testW, Wt, ['phi0.06', 'phim0.06'])
                rs.append([o['phi0.06']['share']['est'], o['phim0.06']['share']['est']])
            D['random_share_plus_minus'] = rs
            R['das'][site] = D
            f = lambda d: f'share {d["share"]["est"]:+.3f} [{d["share"]["ci"][0]:+.3f}, {d["share"]["ci"][1]:+.3f}] IIA {d["iia"]["est"]:+.3f} r {d["r"]["est"] if d["r"]["est"] is None else round(d["r"]["est"], 3)} ds {d["mean_ds"]:+.3f}'
            log(f'   {site} pos-fit: ' + ' | '.join(f'{e} {f(D["pos_fit"][e])}' for e in EV))
            log(f'   {site} both-fit (|cos| {D["both_fit"]["abs_cos_with_pos_fit"]:.3f}): ' + ' | '.join(f'{e} {f(D["both_fit"][e])}' for e in ['phi0.06', 'phim0.06', 'eta0.06', 'etam0.06'])
                + f' | random (+, -) {np.round(rs, 3).tolist()} ({time.time() - t2:.0f} s)')
            res['models'][key] = R; json.dump(P.clean(res), open(FN, 'w'), indent=1)
        R['holm_minus_vs_plus'] = dict(zip([s for s, _ in tests], P.holm([p for _, p in tests])))
        P.DAS_CF = POS_CF
        g = P.graph_block(model, XW['base'], XW, testW, yW, LW['base'], Wt)
        R['graphs'] = {e: dict(edges_changed=g[e]['edges_changed_per_layer'], share={k: dict(est=v['share_dd']['est'], ci=v['share_dd']['ci']) for k, v in g[e]['variants'].items()})
                       for e in P.GRAPH_EDITS}
        for e in P.GRAPH_EDITS:
            log(f'   graphs {e}: edges changed {np.round(R["graphs"][e]["edges_changed"], 4).tolist()}; ' + '; '.join(f'{k} {v["est"]:.3f} {np.round(v["ci"], 3).tolist()}' for k, v in R['graphs'][e]['share'].items()))
        R['atlas_pool'] = atlas_ctrl(RA['pool'], OBS_ALL, y, fit_all, test_all, np.random.default_rng(75))
        log(f'   atlas: {R["atlas_pool"]}')
        circ, _ = P.circuit_block(model, Xall, WZ[fitW], WZ[testW], test_all[::2], y, J0['jet_pt'].astype(np.float64), np.random.default_rng(85), Lall)
        C = circ['pool']
        R['circuits_pool'] = dict(full=C['full'], all_resampled=C['all_resampled'], n_live=C['n_live'],
                                  K={K: dict(wz_acc=v['wz_acc'], wz_acc_ci=v['wz_acc_ci'], acc10=v['acc10'], F=v['F'], mcnemar=v['mcnemar_vs_full'],
                                             random_wz_acc=dict(mean=v['random_wz_acc']['mean'], q95=v['random_wz_acc']['q95'], p_perm=v['random_wz_acc']['p_perm'])) for K, v in C['K'].items()})
        res['models'][key] = R; json.dump(P.clean(res), open(FN, 'w'), indent=1); log(f'{key} done ({time.time() - t1:.0f} s)')
        del model, Xall, XW
        if P.DEV == 'cuda': torch.cuda.empty_cache()
    json.dump(P.clean(res), open(FN, 'w'), indent=1); log(f'wrote {FN}; P15 VERIFY2 DONE')


if __name__ == '__main__':
    main()
