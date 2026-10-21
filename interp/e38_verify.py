"""E38 verification (job e38-verify, 1 CPU / 2 GiB).

Part A, fresh sample(600, 3821) minus jets of E38's main sample(1000, 3801); f = 0.2, all 42 cells of E38.
  T  : E38's edit (E38.transfer_u + E38.make_edit: min-norm angle projection, unweighted in angle units).
  null: E38's soft-sector null (E38.soft_null_u, same GeV L2, same projection).
  pt_only / angle_only: E38's decomposition.
  EQ : same pT transfer, angle projection with equal-angle weighting (weighted min-norm, weight pt^-1/2 per particle, so
       every particle's angle moves by a similar amount instead of the hardest particles moving most).  Tests whether the
       map depends on the choice of projection metric.
  PS : (single-rank cells only, k, j in 0..5) same pT transfer, angles NOT changed; the jet four-momentum is restored by
       min-norm changes of log pT of the rank 6-63 particles only.  A hardness edit with every angle fixed.
       Its compensation alone (PS_comp: only the rank 6-63 pT changes, no transfer) is the matched control.
  Metrics: W-Z margin change on W+Z jets, on Z and on W jets; q-g margin change on q+g jets; call change; mean change
  of pT^D (sd) and M2 (sd); 1000-sample jet bootstrap; McNemar T vs null with Holm.
Part B, E36 stored minimal flips (e36_edits.npz): margin removed by holding S20+S51+S43, S1+S29+S30, each single live
  S unit, and all C(19,3) - 1 = 968 other live-S triples (not only norm-matched ones); per flip group.
Writes $AI_RESULTS/jedi/e38_verify.json.   Run: python -m interp.e38_verify [--pc 600]
"""
import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('KERAS_BACKEND', 'jax')
from interp import e36_counterfactual_flip_jedi as E  # noqa: E402
from interp import e38_hardness_channel_jedi as M  # noqa: E402

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'
T0 = time.time(); G_, Q_, W_, Z_, T_ = range(5)
F = 0.2


def log(*a): print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def project_gen(x0, ut, wfree, P0, ramp=2, iters=12):
    """restore sum p = P0 after the fixed log-pT change ut by a weighted min-norm change of the free coordinates.
    wfree (J, n, 3): per-coordinate weight (0 = fixed); minimises sum (d/w)^2.  -> v (J, n, 3), good"""
    real = x0[..., 0] > 0; J = len(x0); s = np.maximum(np.hypot(P0[:, 0], P0[:, 1]), 1e-9)[:, None]
    wf = (wfree * real[..., None]).reshape(J, 1, -1); delta = np.zeros(x0.shape); good = np.ones(J, bool)
    ut = np.where(real, ut, 0.0)
    for k in range(1, ramp + 1):
        base = np.zeros(x0.shape); base[..., 0] = ut * k / ramp
        for _ in range(iters):
            with np.errstate(all='ignore'):
                v = base + delta; r = (E.fourmom(x0, v) - P0) / s; r0 = np.sqrt((r ** 2).sum(1))
                act = np.flatnonzero(~(r0 < 1e-13))
                if not len(act): break
                A = E.jac(x0[act], v[act]) / s[act][..., None] * wf[act]
                AA = A @ np.transpose(A, (0, 2, 1)); ra = r[act]
                fin = np.isfinite(AA).all((1, 2)) & np.isfinite(ra).all(1); good[act[~fin]] = False
                AA[~fin] = np.eye(4); ra[~fin] = 0; A[~fin] = 0
                d = -(np.einsum('jkn,jk->jn', A, np.einsum('jkl,jl->jk', np.linalg.pinv(AA, rcond=1e-12), ra)) * wf[act][:, 0]).reshape((len(act),) + x0.shape[1:])
                d[~np.isfinite(d).all((1, 2))] = 0
                step = 1.0; pend = np.arange(len(act))
                for _h in range(7):
                    ii = act[pend]; trial = delta[ii] + step * d[pend]
                    rt = np.sqrt((((E.fourmom(x0[ii], base[ii] + trial) - P0[ii]) / s[ii]) ** 2).sum(1))
                    better = np.isfinite(rt) & (rt < r0[ii]); delta[ii[better]] = trial[better]; pend = pend[~better]; step /= 2
                    if not len(pend): break
    v = base + delta; good &= np.isfinite(v).all((1, 2)); v[~good] = 0
    return np.where(real[..., None], v, 0.0), good


def apply_checked(x0, v, good, use):
    xe, _ = E.apply_edit(x0, v); dm, dax, dpt = E.check_constraints(x0, xe)
    return xe, good & use & (dm < 1e-3) & (dax < 1e-3) & (dpt < 1e-3)


def main(a):
    from interp.jedi_core import Jedi, sample
    from interp.e11_mars_all import indices
    from interp.x10_das_jedi import clean
    OUT.mkdir(parents=True, exist_ok=True); JS = OUT / 'e38_verify.json'
    x, y, _ = sample(a.pc, 3821); ii = indices(a.pc, 3821); keep = ~np.isin(ii, indices(1000, 3801))
    x, y = np.asarray(x[keep]), y[keep]
    net = Jedi(64); _run, _rjw = net.run, net.run_jetwise
    net.run = lambda x_, override=None, keep=('logits',), batch=1000: _run(x_, override, keep, batch)
    net.run_jetwise = lambda x_, override, keep=('logits',), batch=1000: _rjw(x_, override, keep, batch)
    R = dict(node='E38-verify', sample=f'sample({a.pc}, 3821) minus E38 main', N=int(len(y)), f=F)
    save = lambda: JS.write_text(json.dumps(clean(R), indent=1))
    x0 = np.asarray(x, np.float64); J = len(x0); P0 = E.p4(x0).sum(1); jpt = np.hypot(P0[:, 0], P0[:, 1])
    real = x0[..., 0] > 0; use = real.sum(1) >= M.NMIN
    S_ = net.run(x, keep=('logits', 'S')); L0, S0 = S_['logits'], S_['S']; c0 = L0.argmax(1); sdS = S0.std(0) + 1e-9
    wz0, qg0 = L0[:, W_] - L0[:, Z_], L0[:, Q_] - L0[:, G_]
    R['acc_exact'] = float((c0 == y).mean()); R['n_used'] = int(use.sum())
    log(f'N {J}, accuracy {R["acc_exact"]:.4f}, used {use.sum()}')
    ph0 = M.physics(x0, jpt); sdp = {k: v.std() for k, v in ph0.items()}
    mWZ, mQG, mZ, mW = (y == W_) | (y == Z_), (y == Q_) | (y == G_), y == Z_, y == W_
    boot = M.Boot(J, a.B, 3822); rng = np.random.default_rng(3823)
    wEQ = np.ones(x0.shape); wEQ[..., 0] = 0; wEQ[..., 1:] *= (np.maximum(x0[..., :1], 1e-9) / np.median(x0[real][:, 0])) ** -0.5
    UNITS = [20, 51, 43, 1, 29, 30]

    def score(xe, ok, lab, cell, store):
        r = net.run(xe.astype(np.float32), keep=('logits', 'S')); L = r['logits']
        dw = np.where(ok, L[:, W_] - L[:, Z_] - wz0, np.nan); dq = np.where(ok, L[:, Q_] - L[:, G_] - qg0, np.nan)
        ch = np.where(ok, (L.argmax(1) != c0).astype(float), np.nan)
        ph = M.physics(xe, jpt)
        st = store.setdefault(lab, dict(wz=[], qg=[], chg=[], comp=[], dptD=[], dM2=[], dzl=[], dS={f'S{u}': [] for u in UNITS}, cells=[]))
        st['wz'].append(dw); st['qg'].append(dq); st['chg'].append(ch); st['comp'].append(float(ok.mean())); st['cells'].append(cell)
        st['dptD'].append(float(np.nanmean(np.where(ok, (ph['ptD'] - ph0['ptD']) / sdp['ptD'], np.nan))))
        st['dM2'].append(float(np.nanmean(np.where(ok, (ph['M2'] - ph0['M2']) / sdp['M2'], np.nan))))
        st['dzl'].append(float(np.nanmean(np.where(ok, (ph['zlead'] - ph0['zlead']) / sdp['zlead'], np.nan))))
        for u in UNITS: st['dS'][f'S{u}'].append(float(np.nanmean(np.where(ok, (r['S'][:, u] - S0[:, u]) / sdS[u], np.nan))))

    store = {}; extra = dict(ang_T=[], ang_EQ=[], ps_comp_rel=[], ps_comp_l2_ratio=[])
    for k, j in M.CELLS:
        cell = M.cname(k, j, F); u, dpt = M.transfer_u(x0, k, j, F)
        xe, v, comp, _ = M.make_edit(x0, u, P0); comp &= use; score(xe, comp, 'T', cell, store)
        ang = E.SC * np.hypot(v[..., 1], v[..., 2]); w = x0[..., 0] * real
        extra['ang_T'].append(float(np.nanmedian((ang * w).sum(1) / w.sum(1) * 1e3)))
        for lab, vv in (('pt_only', np.concatenate([u[..., None], np.zeros(x0.shape[:2] + (2,))], -1)),
                        ('angle_only', np.concatenate([np.zeros(x0.shape[:2] + (1,)), v[..., 1:]], -1))):
            xd, _ = E.apply_edit(x0, vv); score(xd, comp, lab, cell, store)
        un, _ = M.soft_null_u(x0, dpt, rng); xn, _, cn, _ = M.make_edit(x0, un, P0); score(xn, cn & use, 'null', cell, store)
        ve, ge = project_gen(x0, u, wEQ, P0); xq, cq = apply_checked(x0, ve, ge, use); score(xq, cq, 'EQ', cell, store)
        ang = E.SC * np.hypot(ve[..., 1], ve[..., 2]); extra['ang_EQ'].append(float(np.nanmedian((ang * w).sum(1) / w.sum(1) * 1e3)))
        if k != 'rest' and j != 'rest':
            wps = np.zeros(x0.shape); wps[:, 6:, 0] = 1.0
            vp, gp = project_gen(x0, u, wps, P0); xp, cp = apply_checked(x0, vp, gp, use)
            # sanity: angles unchanged, ranks 0-5 pT change = the transfer only
            assert np.abs(vp[..., 1:]).max() == 0 and np.allclose(vp[gp][:, :6, 0], u[gp][:, :6])
            score(xp, cp, 'PS', cell, store)
            comp_u = vp[..., 0].copy(); comp_u[:, :6] = 0
            xc, _ = E.apply_edit(x0, np.concatenate([comp_u[..., None], np.zeros(x0.shape[:2] + (2,))], -1)); score(xc, cp, 'PS_comp', cell, store)
            dg = x0[..., 0] * np.expm1(comp_u) * real
            extra['ps_comp_rel'].append(float(np.nanmedian(np.where(cp, (np.abs(np.expm1(comp_u)) * w).sum(1) / np.maximum((w * (np.arange(64) >= 6)).sum(1), 1e-9), np.nan))))
            extra['ps_comp_l2_ratio'].append(float(np.nanmedian(np.where(cp, np.sqrt((dg ** 2).sum(1)) / np.maximum(np.sqrt((dpt ** 2).sum(1)), 1e-9), np.nan))))
        log(f'{cell}: T {np.nanmean(store["T"]["wz"][-1][mWZ]):+.3f}  EQ {np.nanmean(store["EQ"]["wz"][-1][mWZ]):+.3f}'
            + (f'  PS {np.nanmean(store["PS"]["wz"][-1][mWZ]):+.3f} (comp {np.nanmean(store["PS_comp"]["wz"][-1][mWZ]):+.3f}, ok {store["PS"]["comp"][-1]:.3f})' if 'PS' in store and store['PS']['cells'][-1] == cell else '')
            + f'  null {np.nanmean(store["null"]["wz"][-1][mWZ]):+.3f}')
    A = {}
    for lab, st in store.items():
        WZ, QG, CH = (np.stack(st[kk], 1) for kk in ('wz', 'qg', 'chg'))
        A[lab] = dict(cells=st['cells'], compliance=st['comp'], wz_on_WZ=boot.mean(WZ, mWZ), wz_on_Z=boot.mean(WZ, mZ), wz_on_W=boot.mean(WZ, mW),
                      qg_on_qg=boot.mean(QG, mQG), call_change=boot.mean(CH), dptD_sd=st['dptD'], dM2_sd=st['dM2'], dzlead_sd=st['dzl'], dS_sd=st['dS'])
        st['_WZ'], st['_QG'], st['_CH'] = WZ, QG, CH
    # T - null, McNemar
    Tn = store['T']; Nn = store['null']
    dw = boot.mean(Tn['_WZ'] - Nn['_WZ'], mWZ); dq = boot.mean(Tn['_QG'] - Nn['_QG'], mQG)
    mc = []
    for i in range(len(M.CELLS)):
        ok = np.isfinite(Tn['_CH'][:, i]) & np.isfinite(Nn['_CH'][:, i]); mc.append(M.mcnemar(Tn['_CH'][ok, i] > 0, Nn['_CH'][ok, i] > 0))
    for m, p in zip(mc, M.holm([m['p'] for m in mc])): m['p_holm'] = float(p)
    A['T_minus_null'] = dict(wz_on_WZ=dw, qg_on_qg=dq, n_neg_wz=int((dw['mean'] < 0).sum()), n_neg_qg=int((dq['mean'] < 0).sum()), mcnemar=mc,
                             n_null_more_calls=int(sum(m['second_only'] > m['first_only'] for m in mc)), n_mcnemar_holm05=int(sum(m['p_holm'] < .05 for m in mc)))
    # antisymmetry tests for single-rank pairs: d(k->j) - d(j->k) and d(k->j) + d(j->k)
    pairs = [(k, j) for k in range(6) for j in range(6) if k < j]
    for lab in ('T', 'EQ', 'PS', 'pt_only', 'PS_comp'):
        st = store[lab]; idx = {c: i for i, c in enumerate(st['cells'])}; anti = {}
        for k, j in pairs:
            a_, b_ = st['_WZ'][:, idx[M.cname(k, j, F)]], st['_WZ'][:, idx[M.cname(j, k, F)]]
            anti[f'{k}<->{j}'] = dict(fwd=boot.mean(a_, mWZ), rev=boot.mean(b_, mWZ), sum=boot.mean(a_ + b_, mWZ))
        A.setdefault('pairs_wz', {})[lab] = anti
        md = np.array([np.nanmean(st['_WZ'][mWZ, i]) for i in range(len(st['cells']))]); pd = np.array(st['dptD'])
        A.setdefault('corr_cell_wz_vs_dptD', {})[lab] = float(np.corrcoef(md, pd)[0, 1])
        A.setdefault('n_cells_wz_positive', {})[lab] = [int((md > 0).sum()), int(len(md))]
    # map similarity across variants (W-Z cell means, single-rank cells)
    sr = [i for i, c in enumerate(store['T']['cells']) if 'rest' not in c]
    cm = {lab: np.array([np.nanmean(store[lab]['_WZ'][mWZ, i]) for i in range(len(store[lab]['cells']))]) for lab in store}
    A['corr_maps_single_rank'] = {f'{p}~{q}': float(np.corrcoef(cm[p][sr] if len(cm[p]) == 42 else cm[p], cm[q][sr] if len(cm[q]) == 42 else cm[q])[0, 1])
                                  for p, q in (('T', 'EQ'), ('T', 'PS'), ('PS', 'pt_only'), ('T', 'pt_only'), ('T', 'angle_only'), ('PS', 'PS_comp'))}
    A['angles_mrad_pTweighted_median'] = dict(T=extra['ang_T'], EQ=extra['ang_EQ'])
    A['PS_compensation'] = dict(median_pTweighted_rel_change_rest=extra['ps_comp_rel'], median_L2_GeV_ratio_to_transfer=extra['ps_comp_l2_ratio'])
    R['partA'] = A; save(); del store
    # ---------------- Part B
    k = np.load(OUT / 'e36_edits.npz'); xb = k['x0'].astype(np.float64); vb = k['v'].astype(np.float64)
    ok, prim, tgt = k['ok'], k['prim'], k['tgt']
    b = net.run(xb.astype(np.float32), keep=('logits', 'S')); Lb, Sb = b['logits'], b['S']; cur = Lb.argmax(1)
    xe, _ = E.apply_edit(xb, vb); xe32 = xe.astype(np.float32); e = net.run(xe32, keep=('logits', 'S')); LE, SE = e['logits'], e['S']
    fl = ok & prim & (LE.argmax(1) == tgt); idx = np.flatnonzero(fl); xe32 = xe32[idx]; tg, cu = tgt[idx], cur[idx]
    r_ = np.arange(len(idx)); m0 = Lb[idx][r_, tg] - Lb[idx][r_, cu]; mE = LE[idx][r_, tg] - LE[idx][r_, cu]; S0f = Sb[idx]
    grp = {'W->Z': cu == W_, 'Z->W': cu == Z_, 'q->g': cu == Q_, 'g->q': cu == G_}
    bb = M.Boot(len(idx), a.B, 3824)
    def removed(units):
        m = np.zeros(64, np.float32); m[list(units)] = 1
        L = net.run_jetwise(xe32, {'S': (m, S0f)}, keep=('logits',))['logits']; mh = L[r_, tg] - L[r_, cu]
        return mh, (L.argmax(1) != tg)
    B = dict(n_flips=int(len(idx)), d_margin=bb.mean(mE - m0))
    sets = {}
    for lab, units in (('S20+S51+S43', M.NAMED), ('S1+S29+S30', M.RADIAL)):
        mh, und = removed(units); s = dict(margin_removed=bb.ratio(mE - mh, mE - m0), flips_undone=float(und.mean()), by_group={})
        for g, mm in grp.items():
            if mm.sum(): s['by_group'][g] = dict(n=int(mm.sum()), margin_removed=bb.ratio(mE - mh, mE - m0, mm), flips_undone=float(und[mm].mean()))
        sets[lab] = s; log(f'E36 hold {lab}: removed {s["margin_removed"]}; groups ' + ', '.join(f'{g} {v["margin_removed"]["value"]:.3f}' for g, v in s['by_group'].items()))
    B['sets'] = sets
    single = {}
    for u in M.LIVE_S:
        mh, und = removed([u]); single[f'S{u}'] = dict(margin_removed=float((mE - mh).mean() / (mE - m0).mean()), flips_undone=float(und.mean()),
                                                      by_group={g: float((mE - mh)[mm].mean() / (mE - m0)[mm].mean()) for g, mm in grp.items() if mm.sum()})
    B['single_units'] = single; log('single units: ' + ', '.join(f'{k_} {v["margin_removed"]:.3f}' for k_, v in single.items()))
    trip = [t for t in itertools.combinations(M.LIVE_S, 3) if set(t) != set(M.NAMED)]; rr = []; rg = {g: [] for g in grp}
    for t in trip:
        mh, _ = removed(t); rr.append((mE - mh).mean() / (mE - m0).mean())
        for g, mm in grp.items(): rg[g].append((mE - mh)[mm].mean() / (mE - m0)[mm].mean())
    rr = np.array(rr); nv = sets['S20+S51+S43']['margin_removed']['value']; rv = sets['S1+S29+S30']['margin_removed']['value']
    B['all_triples'] = dict(n=len(trip), q05_q50_q95=np.quantile(rr, [.05, .5, .95]).tolist(),
                            rank_p_named=float((1 + (rr >= nv).sum()) / (1 + len(rr))), rank_p_radial=float((1 + (rr >= rv).sum()) / (1 + len(rr))),
                            by_group_q50_q95={g: np.quantile(np.array(v), [.5, .95]).tolist() for g, v in rg.items()},
                            by_group_rank_p_named={g: float((1 + (np.array(rg[g]) >= sets['S20+S51+S43']['by_group'][g]['margin_removed']['value']).sum()) / (1 + len(rr))) for g in grp if g in sets['S20+S51+S43']['by_group']})
    log(f'all triples: {B["all_triples"]}')
    R['partB'] = B; save(); log(f'wrote {JS}; done')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--pc', type=int, default=600); ap.add_argument('--B', type=int, default=1000)
    main(ap.parse_args())
