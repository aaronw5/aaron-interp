"""P15 verification (loop C): rerun the key P15 parts on a fresh sample and add the missing controls.

Fresh sample: JetClass val_5M <sample>_122.root (P15 used _120), 2,000 random rows per class (numpy default_rng(25)), fit/test
halves from default_rng(26); same builds, edits and functions as interp/p15_particlenet_internal.py (imported, not copied).
Parts rerun: edit response; one-direction DAS per site (ec1, ec2, ec3 = pooled, fc) with 5 random directions and the weight-
shuffled network; P8v slope ratios and edit regression; graph freezes (phi0.06); literal pooled-channel circuits (K = 8..64,
100 random sets, W/Z and 10-class accuracy).
Added controls:
  (a) DAS restarted from a second random init (cosine between the two directions; share of the second);
  (b) the untrained mean-difference direction q = normalise(mean_fit (h_phi0.06 - h)) at each site (no optimisation);
  (c) the weight-shuffled network with its BatchNorm statistics re-estimated on these jets (perm_bn) and a freshly initialised
      ParticleNet with re-estimated BatchNorm statistics (reinit_bn): the P15 shuffled network keeps the trained BN statistics,
      so its logits are 20-45x larger than the trained network's; these two controls have data-scaled activations.
  For every network: share of the phi0.06 change in d reproduced, 10-logit IIA, per-jet Pearson r(dd_int, dd_cf).
  Bootstrap (1,000 over test jets) of trained-minus-control share.
  ./sync.sh && ./part_job.sh c-p15-verify 'python -m interp.p15_verify'   -> $AI_RESULTS/part/p15_verify.json
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import interp.p15_particlenet_internal as P

P.SEED = 25
P.LEVELS = ['pool']
P.GRAPH_EDITS = ['phi0.06']
FILE_ID = 122
OUTV = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
FN = OUTV / 'p15_verify.json'
T0 = time.time()


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


def recal_bn(model, Xt, n_rows, rng, nb=20, B=500):
    """re-estimate every BatchNorm's running statistics on data (cumulative average over nb batches), weights untouched"""
    torch = P.torch
    for mod in model.modules():
        if isinstance(mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)): mod.reset_running_stats(); mod.momentum = None
    model.train()
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout): mod.eval()
    with torch.no_grad():
        for _ in range(nb):
            b = torch.as_tensor(rng.choice(n_rows, B, replace=False), device=P.DEV)
            model(Xt['p'][b].clone(), Xt['x'][b].clone(), Xt['m'][b])
    return P.freeze(model)


def per_net_stats(model, XWb, site, q, MW, LW, testW, Wt):
    ev = P.das_eval(model, XWb, site, q, MW, LW, testW, ['phi0.06']); st, B = P.das_stats(ev, Wt)
    e = ev['phi0.06']; r, _ = P.wcorr(Wt, e['dd_int'], e['dd_cf'])
    s = st['phi0.06']
    return dict(share=s['share_dd'], iia=s['iia'], call_iia=s['call_iia'], n_changed=s['n_changed'], mean_dd_cf=s['mean_dd_cf'],
                mean_abs_dd_cf=float(np.abs(e['dd_cf']).mean()), r_dd=r), B['phi0.06']['share']


def main():
    P._imports(); torch = P.torch; pc = P.pc; p4 = P.p4; pn_net = P.pn_net; FEATURES = P.FEATURES
    P.DEV = 'cuda' if torch.cuda.is_available() else 'cpu'; OUTV.mkdir(parents=True, exist_ok=True)
    PNDIR = Path(os.environ.get('JETDISTILL_MODELS', '/j-jepa-vol/aaron-interp/models/part')).parent / 'particlenet'
    res = dict(file_id=FILE_ID, seed=25, per_class=P.PER, n_boot=P.NB, steps=P.STEPS, device=P.DEV, models={})
    rng = np.random.default_rng(25); parts = []
    for c in pc.CL:
        J, r, N = load_class(c, rng); parts.append(J); log(f'loaded {c}: {len(r)} of {N}')
    J0 = pc.cat(parts); del parts; y = J0['label'].astype(int); N = len(y)
    pt, de, dp, ms = pc.kin(J0); real = (pt > 0).astype(np.float32); Jb = pc.build(J0, pt, de, dp, ms)
    perm = np.random.default_rng(26).permutation(N); is_test = np.zeros(N, bool); is_test[perm[N // 2:]] = True
    iW, iZ = P.iW, P.iZ
    WZ = np.where((y == iW) | (y == iZ))[0]; fitW = np.where(~is_test[WZ])[0]; testW = np.where(is_test[WZ])[0]
    test_all = np.where(is_test)[0]
    res['sample'] = dict(N=N, n_wz=len(WZ), n_wz_fit=len(fitW), n_wz_test=len(testW))
    JW0 = pc.sub(J0, WZ); yW = y[WZ]; ptW, deW, dpW, msW = pt[WZ], de[WZ], dp[WZ], ms[WZ]; rW = real[WZ]
    JEW = {'base': pc.sub(Jb, WZ)}; OBW = {'base': P.basic_obs(JEW['base'])}
    for dd_ in (0.03, 0.06, 0.09): JEW[f'phi{dd_}'] = pc.build(JW0, ptW, deW, dpW + dd_ * rW, msW)
    JEW['eta0.06'] = pc.build(JW0, ptW, deW + 0.06 * rW, dpW, msW)
    OBW['phi0.06'] = P.basic_obs(JEW['phi0.06']); kmatch = float(np.exp((OBW['phi0.06']['log_mA'] - OBW['base']['log_mA']).mean()))
    JEW['angk'] = pc.build(JW0, ptW, deW * kmatch, dpW * kmatch, msW); JEW['ang1.1'] = pc.build(JW0, ptW, deW * 1.1, dpW * 1.1, msW)
    JEW['pt1.25'] = pc.build(JW0, ptW * 1.25, deW, dpW, msW * 1.25)
    a = 0.3; JEW['rot0.3'] = pc.build(JW0, ptW, (deW * np.cos(a) - dpW * np.sin(a)) * rW, (deW * np.sin(a) + dpW * np.cos(a)) * rW, msW)
    for e in P.EDITS: OBW[e] = P.basic_obs(JEW[e]); assert np.array_equal(JEW[e]['mask'], JEW['base']['mask']), e
    res['edits'] = dict(k_match=kmatch, checks={e: dict(dlogm=float((OBW[e]['log_m'] - OBW['base']['log_m']).mean()),
                                                        dlogmA=float((OBW[e]['log_mA'] - OBW['base']['log_mA']).mean())) for e in P.EDITS})
    log(f'sample N {N}, W/Z {len(WZ)} (fit {len(fitW)}, test {len(testW)}), k_match {kmatch:.4f}; edits {res["edits"]["checks"]}')
    Wt = P.bootW(len(testW), 28); dfun = lambda L: L[:, iW] - L[:, iZ]
    for n in ('full', 'kin'):
        t1 = time.time(); key = f'ParticleNet_{n}'; nf = len(FEATURES[n]); R = {}
        model = P.freeze(pn_net.load_particlenet(PNDIR / f'ParticleNet_{n}.pt', nf, P.DEV))
        Xall = P.to_dev(p4.pn_inputs(Jb, n)); RA = P.fset(model, Xall, np.arange(N)); Lall = RA['L']
        R['acc'] = float((Lall.argmax(1) == y).mean())
        XW = {e: P.to_dev(p4.pn_inputs(JEW[e], n)) for e in ['base'] + P.EDITS}
        RW = {e: P.fset(model, XW[e], np.arange(len(WZ))) for e in ['base'] + P.EDITS}
        LW = {e: RW[e]['L'] for e in RW}; MW = {e: {s: P.site_val(RW[e], s) for s in P.SITES} for e in RW}
        R['W_to_Z_base'] = float((LW['base'].argmax(1)[yW == iW] == iZ).mean()); R['W_to_Z_phi0.06'] = float((LW['phi0.06'].argmax(1)[yW == iW] == iZ).mean())
        R['mean_dd_phi0.06_test'] = float((dfun(LW['phi0.06']) - dfun(LW['base']))[testW].mean())
        log(f'{key}: acc {100 * R["acc"]:.2f} %; W->Z {R["W_to_Z_base"]:.3f} -> {R["W_to_Z_phi0.06"]:.3f}; mean dd test {R["mean_dd_phi0.06_test"]:+.3f}')
        # control networks
        ctrl = {}
        ctrl['perm'] = P.freeze(P.permuted_model(PNDIR / f'ParticleNet_{n}.pt', nf))
        ctrl['perm_bn'] = recal_bn(P.permuted_model(PNDIR / f'ParticleNet_{n}.pt', nf), Xall, N, np.random.default_rng(41))
        torch.manual_seed(25); m0 = pn_net.ParticleNet(input_dims=nf, **pn_net.PN_CFG).to(P.DEV)
        ctrl['reinit_bn'] = recal_bn(m0, Xall, N, np.random.default_rng(42))
        CW = {}
        for cn, cm in ctrl.items():
            RWc = {e: P.fset(cm, XW[e], np.arange(len(WZ))) for e in ['base'] + P.DAS_CF}
            CW[cn] = ({e: {s: P.site_val(RWc[e], s) for s in P.SITES} for e in RWc}, {e: RWc[e]['L'] for e in RWc})
            Lb = CW[cn][1]['base']; R[f'{cn}_acc_wz'] = float((Lb.argmax(1) == yW).mean())
            R[f'{cn}_mean_abs_d'] = float(np.abs(dfun(Lb)).mean())
        R['trained_mean_abs_d'] = float(np.abs(dfun(LW['base'])).mean())
        rng_d = np.random.default_rng(35); R['das'] = {}
        for site in P.SITES:
            t2 = time.time(); D = {}
            q, losses = P.das_train(model, XW['base'], site, MW, LW, fitW, rng_d)
            D['trained'], Btr = per_net_stats(model, XW['base'], site, q, MW, LW, testW, Wt)
            ev = P.das_eval(model, XW['base'], site, q, MW, LW, testW, P.EVAL_CF); st, _ = P.das_stats(ev, Wt)
            D['held_out_share'] = {e: st[e]['share_dd']['est'] for e in P.EVAL_CF}
            q2, _ = P.das_train(model, XW['base'], site, MW, LW, fitW, rng_d)
            D['restart'], _ = per_net_stats(model, XW['base'], site, q2, MW, LW, testW, Wt)
            D['restart']['abs_cos_with_first'] = float(abs((q @ q2).item()))
            dm = (MW['phi0.06'][site][fitW] - MW['base'][site][fitW]).mean(0); qm = torch.as_tensor(dm / np.linalg.norm(dm), dtype=torch.float32, device=P.DEV)
            D['meandiff'], Bmd = per_net_stats(model, XW['base'], site, qm, MW, LW, testW, Wt)
            D['meandiff']['abs_cos_with_das'] = float(abs((q @ qm).item()))
            rs = []
            for r in range(P.NRAND):
                qr = torch.randn(P.DIM[site], generator=torch.Generator().manual_seed(2500 + r)).to(P.DEV); qr = qr / qr.norm()
                rs.append(per_net_stats(model, XW['base'], site, qr, MW, LW, testW, Wt)[0]['share']['est'])
            D['random_share'] = rs
            for cn, cm in ctrl.items():
                MWc, LWc = CW[cn]; qc, _ = P.das_train(cm, XW['base'], site, MWc, LWc, fitW, rng_d)
                D[cn], Bc = per_net_stats(cm, XW['base'], site, qc, MWc, LWc, testW, Wt)
                diff = Btr - Bc; D[cn]['trained_minus_ctrl_share'] = dict(est=D['trained']['share']['est'] - D[cn]['share']['est'], ci=P.ci(diff), **P.p_two(diff))
            sr = P.slope_ratios(q, site, MW, testW, OBW, Wt, yW)
            D['P8v'] = {k: dict(est=v['est'], ci=v['ci']) for k, v in sr.items() if k.startswith('P8v')}
            er = sr['edit_regression']; D['edit_regression'] = dict(ratio=er['ratio_A_over_inv'], ratio_ci=er['ratio_ci'], share_A=er['share_A'], share_A_ci=er['share_A_ci'])
            R['das'][site] = D
            f = lambda d: f'share {d["share"]["est"]:.3f} [{d["share"]["ci"][0]:.3f}, {d["share"]["ci"][1]:.3f}] IIA {d["iia"]["est"]:.3f} r {d["r_dd"]["est"] if d["r_dd"]["est"] is None else round(d["r_dd"]["est"], 3)}'
            log(f'   {site}: trained {f(D["trained"])} | restart {f(D["restart"])} cos {D["restart"]["abs_cos_with_first"]:.3f} | meandiff {f(D["meandiff"])} | random {np.round(rs, 3).tolist()} | ' +
                ' | '.join(f'{cn} {f(D[cn])} dd_cf {D[cn]["mean_dd_cf"]:+.2f}' for cn in ctrl) + f' | held-out {D["held_out_share"]} | P8v ' +
                ', '.join(f'{k} {v["est"]:.3f}' for k, v in D['P8v'].items()) + f' | edit-reg ratio {er["ratio_A_over_inv"]:.2f} ({time.time() - t2:.0f} s)')
            res['models'][key] = R; json.dump(P.clean(res), open(FN, 'w'), indent=1)
        del ctrl, CW
        g = P.graph_block(model, XW['base'], XW, testW, yW, LW['base'], Wt)['phi0.06']
        R['graphs'] = dict(edges_changed=g['edges_changed_per_layer'], share={k: dict(est=v['share_dd']['est'], ci=v['share_dd']['ci']) for k, v in g['variants'].items()})
        log(f'   graphs: edges changed {np.round(g["edges_changed_per_layer"], 4).tolist()}; shares ' + '; '.join(f'{k} {v["est"]:.3f}' for k, v in R['graphs']['share'].items()))
        circ, _ = P.circuit_block(model, Xall, WZ[fitW], WZ[testW], test_all[::2], y, J0['jet_pt'].astype(np.float64), np.random.default_rng(65), Lall)
        C = circ['pool']
        R['circuits_pool'] = dict(full=C['full'], all_resampled=C['all_resampled'], n_live=C['n_live'],
                                  K={K: dict(wz_acc=v['wz_acc'], wz_acc_ci=v['wz_acc_ci'], acc10=v['acc10'], F=v['F'], mcnemar=v['mcnemar_vs_full'],
                                             random_wz_acc=dict(mean=v['random_wz_acc']['mean'], q05=v['random_wz_acc']['q05'], q95=v['random_wz_acc']['q95'], p_perm=v['random_wz_acc']['p_perm']),
                                             random_acc10_mean=v['random_acc10']['mean']) for K, v in C['K'].items()})
        res['models'][key] = R; json.dump(P.clean(res), open(FN, 'w'), indent=1); log(f'{key} done ({time.time() - t1:.0f} s)')
        del model, Xall, XW
        if P.DEV == 'cuda': torch.cuda.empty_cache()
    json.dump(P.clean(res), open(FN, 'w'), indent=1); log(f'wrote {FN}; P15 VERIFY DONE')


if __name__ == '__main__':
    main()
