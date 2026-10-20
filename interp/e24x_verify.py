"""E24x verify: fresh-seed rerun of the key E24x numbers and causal tests of the multipole names of S58 / S31 by rigid
(Δη, Δφ) transforms of every particle (rotation 90°, 180°, mirrors) and a pT rescale.

Fresh samples: fit = sample(3000, 4101) (N 15,000); test = sample(1500, 4102) minus jets in the fit sample.
(A) probe exactness, accuracy; (B) arm 0 (particles replaced by a donor's, g held), N 2,000;
(C) arm 1: theta of S58 / S31 named by count-weighted multipole sums (E24x theta_multipole, refit here), by MARS / GBT on
    rotation-invariant jet observables; per-g preg R^2 by block; within-window per-slot l3 quantizer check;
(D) g interchange with random donors (N 3,000): slope of actual dS on the named / structural prediction;
(E) input edits applied to every real particle of 3,000 test jets:
      rot90  (deta, dphi) -> (-dphi, deta)   dipole rotated, quadrupole sign-flipped (cos2a, sin2a -> -cos2a, -sin2a)
      rot180 (deta, dphi) -> (-deta, -dphi)  dipole sign-flipped, quadrupole unchanged
      mirror_eta (deta -> -deta)             sum deta and sum r sin2a flip; sum dphi, sum r cos2a unchanged
      mirror_phi (dphi -> -dphi)             sum dphi and sum r sin2a flip
      pt_x2 (all pT doubled)                 count multipoles unchanged; pT-weighted quantities change
    H0 (theta a rotation-invariant scalar, E24x competitor b): dS = 0 under rot / mirror. H1 (multipole names):
    dS = named prediction (slope 1). Pure-quadrupole reading of S58: |dS58| under rot180 << under rot90 (paired
    bootstrap). Metrics: |dS|/sd(S) with CI; slopes of actual on predicted (named m0+m1(+m2), exact chain, structural);
    path calls (only S58, S31 set to edited values).
(F) untrained redraws (seed 0, 1, 2): per-g preg R^2 by block for the g inputs of the picked units (as E24x).
  ./job.sh e24x-verify 'python -m interp.e24x_verify'  ->  $AI_RESULTS/jedi/e24x_verify.json
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
T0 = time.time()
from interp.e24x_context_jedi import (Probe, Gate, theta_multipole, mp_basis, jet_obs, Mars, r2, r2_ci, slope_ci, ci,
                                      mcnemar, holm, clean, jet_hash, NB, bidx)

UNITS = [58, 31]
SMOKE = '--smoke' in sys.argv


def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def edit(x, kind):
    xe = np.array(x, np.float32, copy=True); e, f = x[..., 1].copy(), x[..., 2].copy()
    if kind == 'rot90': xe[..., 1], xe[..., 2] = -f, e
    elif kind == 'rot180': xe[..., 1], xe[..., 2] = -e, -f
    elif kind == 'mirror_eta': xe[..., 1] = -e
    elif kind == 'mirror_phi': xe[..., 2] = -f
    elif kind == 'pt_x2': xe[..., 0] = 2 * x[..., 0]
    return xe


def paired_p(a, b):
    """two-sided bootstrap p for mean(a - b) = 0 (paired over jets)"""
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64); bb = d[bidx(len(d))].mean(1); bb = bb - bb.mean()
    return float(min(1.0, ((np.abs(bb) >= abs(d.mean())).sum() + 1) / (NB + 1)))


def main():
    from interp.jedi_core import sample, Jedi
    from sklearn.ensemble import HistGradientBoostingRegressor
    rng = np.random.default_rng(4103)
    R = dict(node='E24x-verify', fit_sample='sample(3000, 4101)', test_sample='sample(1500, 4102) minus overlap')
    OUT.mkdir(parents=True, exist_ok=True)
    save = lambda: (OUT / ('e24x_verify_smoke.json' if SMOKE else 'e24x_verify.json')).write_text(json.dumps(clean(R), indent=1))
    x, y, _ = sample(*((300, 4101) if SMOKE else (3000, 4101))); xt, yt, _ = sample(*((200, 4102) if SMOKE else (1500, 4102)))
    hf = set(jet_hash(x).tolist()); keep = np.array([h not in hf for h in jet_hash(xt)]); xt, yt = xt[keep], yt[keep]
    R.update(N_fit=len(x), N_test=len(xt), overlap_removed=int((~keep).sum()))
    j = Jedi(64)
    R['acc_fit'] = float((j.run(x)['logits'].argmax(1) == y).mean()); R['acc_test'] = float((j.run(xt)['logits'].argmax(1) == yt).mean())
    pr = Probe(j, UNITS); D = pr(x); Dt = pr(xt)
    ref = j.run(xt[:2000], keep=('S', 'g'))
    R['probe_exact'] = dict(max_abs_dS=float(np.abs(ref['S'] - Dt['S'][:2000]).max()), max_abs_dg=float(np.abs(ref['g'] - Dt['g'][:2000]).max()))
    sdS = D['S'].std(0) + 1e-12
    R['W3'] = {f'S{u}': dict(nonzero_rows=np.flatnonzero(pr.W3[:, u]).tolist(), w=pr.W3[pr.W3[:, u] != 0, u].tolist(),
                             s_rows_live=int(((pr.W3[:, u] != 0) & (D['sbar'].std(0) > 0)).sum()),
                             max_abs_own=float(np.abs(D['own'][..., q]).max()), b3=float(pr.b3[q])) for q, u in enumerate(UNITS)}
    log('N', len(x), len(xt), 'acc', R['acc_fit'], R['acc_test'], 'probe', R['probe_exact'], json.dumps(clean(R['W3'])))
    save()
    # (B) arm 0
    N0 = min(2000, len(xt)); r0 = rng.choice(len(xt), N0, replace=False); d0 = rng.integers(0, len(xt), N0)
    d0 = np.where(d0 == r0, (d0 + 1) % len(xt), d0)
    Ssw = j.run_jetwise(xt[d0], {'g': (np.ones(64), Dt['g'][r0])}, keep=('S',))['S']; dSw = Ssw - Dt['S'][r0]
    liveS = np.flatnonzero(D['S'].std(0) > 0)
    R['arm0'] = dict(n=N0, **{f'S{u}': float(np.abs(dSw[:, u]).max()) for u in UNITS},
                     other_live_frac_changed_median=float(np.median([(np.abs(dSw[:, k]) > 1e-9).mean() for k in liveS if k not in UNITS])))
    log('arm0', R['arm0']); save()
    # (C) arm 1
    live_g = np.flatnonzero(D['g'].std(0) > 1e-9)
    FjI, nI = jet_obs(x); FjIt, _ = jet_obs(xt)
    models = {}; R['arm1'] = {}
    for q, u in enumerate(UNITS):
        gt = Gate(pr, D, q); th = -(D['ctx'][:, q] + gt.b); tht = -(Dt['ctx'][:, q] + gt.b)
        mp, theta_mp, ks = theta_multipole(pr, D, Dt, x, xt, q, u, live_g)
        mI = Mars(nI, maxt=8).fit(FjI, th, FjIt, tht)
        gb = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(FjI, th)
        win = gt.window
        pre = Dt['pre'][..., q]; prew = pre[:, win]; add = (Dt['ctx'][:, q] + gt.b)[:, None]
        # per-slot l3 quantizer: within-window slope of pre on the additive (ctx + b3) value, per slot
        sl = [float(np.polyfit(add[:, 0], prew[:, k], 1)[0]) for k in range(prew.shape[1])]
        R['arm1'][f'S{u}'] = dict(
            n_window_slots=int(win.sum()), sd_theta=float(th.std()),
            theta_r2=dict(multipole={k: v['theta_r2'] for k, v in mp['levels'].items()}, chain_exact=mp['chain_exact']['theta_r2'],
                          mars_invariant=r2(tht, mI.predict(FjIt)), gbt_invariant=r2(tht, gb.predict(FjIt)),
                          linear_live_g=r2(tht, -(Dt['g'][:, live_g] @ pr.W3[live_g, u] + gt.b))),
            S_r2=dict(multipole={k: v['S_r2'] for k, v in mp['levels'].items()}, chain_exact=mp['chain_exact']['S_r2'],
                      additive_exact_theta=r2_ci(Dt['S'][:, u], gt.S(Dt['own'][..., q], tht)),
                      exact_pre=r2_ci(Dt['S'][:, u], gt.scale * (np.maximum(0, pre) * win[None]).sum(1))),
            preg_r2_by_level={f'g{k}': {L: v['units'][f'g{k}']['preg_r2'] for L, v in mp['levels'].items()} for k in ks},
            top_std_coefs_m2={f'g{k}': mp['levels']['m0+m1+m2']['units'][f'g{k}']['std_coefs'][:5] for k in ks},
            g_units=mp['g_units'],
            per_slot_l3=dict(slope_pre_on_additive_min=min(sl), median=float(np.median(sl)), max=max(sl),
                             within_jet_sd_pre_over_window=float(prew.std(1).mean()), sd_additive=float(add.std())),
            mars_invariant_formula=mI.formula())
        models[u] = dict(gate=gt, theta_mp=theta_mp, ks=ks, q=q)
        log(f'S{u}', json.dumps(clean({k: v for k, v in R['arm1'][f'S{u}'].items() if k not in ('mars_invariant_formula', 'g_units')})))
    save()
    Bt = mp_basis(xt)[0]

    def preds(q, u, Dbase, Bbase, Dnew, Bnew):
        M = models[u]; gt = M['gate']; w = pr.W3[:, u]; ks = M['ks']; own0 = np.zeros_like(Dbase['own'][..., q])
        P = dict(structural=gt.S(own0, -(Dnew['ctx'][:, q] + gt.b)) - gt.S(own0, -(Dbase['ctx'][:, q] + gt.b)),
                 chain_exact=gt.S(own0, -(np.maximum(Dnew['preg'][:, ks], 0) @ w[ks] + gt.b)) - gt.S(own0, -(np.maximum(Dbase['preg'][:, ks], 0) @ w[ks] + gt.b)))
        for L in ('m0+m1', 'm0+m1+m2'):
            P['named_' + L] = gt.S(own0, M['theta_mp'](Bnew, L)) - gt.S(own0, M['theta_mp'](Bbase, L))
        return P
    # (D) random-donor g interchange
    N2 = min(3000, len(xt)); rec = rng.choice(len(xt), N2, replace=False); don = rng.integers(0, len(xt), N2)
    don = np.where(don == rec, (don + 1) % len(xt), don)
    xr = xt[rec]; Dr = {k: v[rec] for k, v in Dt.items()}; Da = pr(xr, g=Dt['g'][don]); Dd = {k: v[don] for k, v in Dt.items()}
    R['interchange_random'] = {}
    for q, u in enumerate(UNITS):
        act = Da['S'][:, u] - Dr['S'][:, u]; P = preds(q, u, Dr, Bt[rec], Dd, Bt[don])
        R['interchange_random'][f'S{u}'] = {k: slope_ci(v, act) for k, v in P.items()}
    log('interchange', json.dumps(clean({u: {k: (v['slope'], v['ci'], v['r2']) for k, v in d.items()} for u, d in R['interchange_random'].items()})))
    save()
    # (E) rigid edits
    base_call = j.run(xr)['logits'].argmax(1); mS = np.zeros(64); mS[UNITS] = 1
    Sdims = (lambda v: v[:, None, :]) if bool(j.L['qs1'].keepdims) else (lambda v: v)
    R['edits'] = {}; absd = {}; ps = []
    for kind in ('rot90', 'rot180', 'mirror_eta', 'mirror_phi', 'pt_x2'):
        xe = edit(xr, kind); De = pr(xe); Be = mp_basis(xe)[0]
        FjIe, _ = jet_obs(xe)
        E = dict(max_abs_change_invariant_obs_rel=float(np.max(np.abs(FjIe - FjIt[rec]) / (np.abs(FjIt[rec]) + 1e-6))),
                 basis_change={n: float(np.abs(Be[:, i] - Bt[rec][:, i]).mean() / (Bt[:, i].std() + 1e-12))
                               for i, n in enumerate(mp_basis(xt[:2])[1]) if n in ('sum_deta', 'sum_dphi', 'sum_rcos2', 'sum_rsin2', 'sum_pT', 'sum_r')},
                 mean_abs_dg_live=float(np.abs(De['g'][:, live_g] - Dr['g'][:, live_g]).mean()))
        Se = Dr['S'].copy(); Se[:, UNITS] = De['S'][:, UNITS]
        cp = j.run_jetwise(xr, {'S': (mS, Sdims(Se))}, keep=('logits',))['logits'].argmax(1) != base_call
        cf = j.run(xe)['logits'].argmax(1) != base_call
        E['path_calls_changed_pct'] = ci(100 * cp); E['full_calls_changed_pct'] = ci(100 * cf)
        for q, u in enumerate(UNITS):
            act = De['S'][:, u] - Dr['S'][:, u]; P = preds(q, u, Dr, Bt[rec], De, Be)
            absd[(kind, u)] = np.abs(act) / sdS[u]
            E[f'S{u}'] = dict(abs_dS_over_sd=ci(np.abs(act) / sdS[u]), dS_over_sd=ci(act / sdS[u]),
                              frac_changed=float((np.abs(act) > 1e-9).mean()),
                              invariant_prediction='0 (rotation-invariant observables unchanged)' if kind != 'pt_x2' else 'nonzero',
                              **{k: slope_ci(v, act) for k, v in P.items()},
                              abs_pred_named_m2_over_sd=ci(np.abs(P['named_m0+m1+m2']) / sdS[u]))
            if kind in ('rot90', 'rot180', 'mirror_eta'):
                Lb = 'm0+m1+m2' if u == 58 else 'm0+m1'
                ps.append((f'{kind}_S{u}_named_{Lb}_slope0', E[f'S{u}']['named_' + Lb]['p_slope0']))
        R['edits'][kind] = E
        log(kind, json.dumps(clean({k: (v if not isinstance(v, dict) or 'mean' in v else {kk: (vv['slope'], vv['ci'], vv['r2']) if isinstance(vv, dict) and 'slope' in vv else vv for kk, vv in v.items()})
                                   for k, v in E.items() if k != 'basis_change'})), json.dumps(clean(E['basis_change'])))
        save()
    # pure-quadrupole test for S58: |dS58| rot180 vs rot90 (paired); same for S31 as a contrast
    R['rot180_vs_rot90'] = {f'S{u}': dict(diff_abs_dS_over_sd=ci(absd[('rot180', u)] - absd[('rot90', u)]),
                                          ratio=float(absd[('rot180', u)].mean() / (absd[('rot90', u)].mean() + 1e-12)),
                                          p=paired_p(absd[('rot180', u)], absd[('rot90', u)])) for u in UNITS}
    adj = holm([p for _, p in ps]); R['holm'] = {k: dict(p=p, holm=float(a)) for (k, p), a in zip(ps, adj)}
    log('rot180 vs rot90', json.dumps(clean(R['rot180_vs_rot90'])), 'holm', json.dumps(clean(R['holm']))); save()
    # (F) untrained redraws: per-g preg R^2 by block
    R['untrained'] = {}
    for seed in ((0,) if SMOKE else (0, 1, 2)):
        ju = Jedi(64, untrained=True, seed=seed); pu = Probe(ju, UNITS); Du = pu(x[:6000]); Dut = pu(xt[:3000])
        lg = np.flatnonzero(Du['g'].std(0) > 1e-9); rs = {}
        for q, u in enumerate(UNITS):
            if Du['S'][:, u].std() == 0: rs[f'S{u}'] = 'dead'; continue
            mp, _, ks = theta_multipole(pu, Du, Dut, x[:6000], xt[:3000], q, u, lg)
            if not ks: rs[f'S{u}'] = 'no live g input'; continue
            rs[f'S{u}'] = dict(theta_r2={k: v['theta_r2'] for k, v in mp['levels'].items()}, chain=mp['chain_exact']['theta_r2'],
                               preg_r2_by_level={f'g{k}': {L: v['units'][f'g{k}']['preg_r2'] for L, v in mp['levels'].items()} for k in ks},
                               sd_own=float(Du['own'][..., q].std()), sd_ctx=float(Du['ctx'][:, q].std()))
        R['untrained'][f'seed{seed}'] = rs; log('untrained', seed, json.dumps(clean(rs))); save(); del ju, pu
    log('done'); save()


if __name__ == '__main__':
    main()
