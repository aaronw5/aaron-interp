"""E37 verification (fresh jets): replicate the h4 / h5 DAS results of E37 on jets not used by E37, and add the controls the
E37 run did not have.
Samples: fit F' = W+Z jets of sample(1500, 3801), test T' = W+Z jets of sample(1500, 3802); every row that is in E37's F
(sample(2000, 3701)) or T (sample(2000, 3702)) is removed from both, and F' rows are removed from T'.
Same edits (E37 prep: D(dc), cn, dil x0.95 / x1.05, cross), same network (JNet, exact fixed-point tail).
 (1) stored E37 subspaces (h4 k=1,2,3; h5 k=1; S k=1) evaluated on T' without refitting: frac, IIA, slope, cn / dil transfer,
     1,000 jet-bootstrap CIs; 300 random live subspaces per k=1 / k=2 (h4).
 (2) DAS refit at h4 k=1 on F' (400 steps, same code) and on deranged targets; |cos| between refit and stored direction.
 (3) non-learned k=1 baselines at h4 (fit on F'): PC1 of the self-pair displacements h(D) - h(unedited) (uncentred);
     mean-displacement direction (mean over dc +0.3 pairs minus mean over dc -0.3 pairs); margin read-out direction
     (least squares of the W-Z margin on h4 across unedited F' jets).
 (4) positive control for the dilation-transfer metric: the stored M2 (k=2) DAS subspace and the stored dilation direction at h4,
     evaluated on dilation pairs and on D pairs.
 (5) steering on T' along the stored h4 k=1 direction (E37 steered only along the k=2 read-out direction) and along the
     k=2 read-out direction, 100 random live directions each.
Writes $AI_RESULTS/jedi/e37_verify.json.  Run: ./job.sh e37-verify 'python -m interp.e37_verify'
"""
import json
import os
import time
from pathlib import Path

import numpy as np

from interp import e37_sharing_das_jedi as E

RES = Path(os.environ.get('AI_RESULTS', Path(__file__).resolve().parents[1] / 'results')); OUT = RES / 'jedi'
VF, VT, PER = 3801, 3802, 1500
T0 = time.time()


def log(*a): print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def ev(net, st, Q, PT, W, sets=('self', 'cross', 'cn', 'dil')):
    out = {}
    for nm in sets:
        lp = E.patch(net, st, PT[nm].hb, PT[nm].hs, Q); m = E.metrics(lp, PT[nm]); m.update(E.metrics(lp, PT[nm], W)); out[nm] = m
    return out


def rand_null(net, st, PT, live, d, k, n, seed):
    rr = np.random.default_rng(seed); R = {nm: dict(iia=[], frac=[]) for nm in ('self', 'dil')}
    cat = [PT['self'], PT['dil']]; HB = np.concatenate([p.hb for p in cat]); HS = np.concatenate([p.hs for p in cat]); c = len(PT['self'].mt)
    for _ in range(n):
        Qr = E.rand_live(rr, live, d, k); lp = E.patch(net, st, HB, HS, Qr)
        for nm, sl in (('self', slice(0, c)), ('dil', slice(c, None))):
            m = E.metrics(lp[sl], PT[nm]); R[nm]['iia'].append(np.nan if m['iia_wz'] is None else m['iia_wz']); R[nm]['frac'].append(m['frac'])
    return {nm: dict(iia=np.asarray(v['iia'], float), frac=np.asarray(v['frac'], float)) for nm, v in R.items()}


def summ_null(N, o):
    out = {}
    for nm in N:
        a, f = N[nm]['iia'], N[nm]['frac']; ob = o[nm]
        out[nm] = dict(iia_mean=float(np.nanmean(a)), iia_q95=float(np.nanquantile(a, .95)), frac_mean=float(np.nanmean(f)),
                       frac_q=np.nanquantile(f, [.025, .5, .95, .975]).tolist(),
                       p_iia=float((1 + (a[np.isfinite(a)] >= ob['iia_wz']).sum()) / (1 + np.isfinite(a).sum())) if ob['iia_wz'] is not None else None,
                       p_frac=float((1 + (np.abs(f) >= abs(ob['frac'])).sum()) / (1 + len(f))))
    return out


def short(m):
    return {k: m.get(k) for k in ('frac', 'frac_ci', 'iia_wz', 'iia_wz_ci', 'n_flip', 'slope', 'slope_ci', 'mean_abs_patch', 'mean_abs_target')}


def main():
    import jax
    from interp.jedi_core import sample
    from interp.x10_das_jedi import JNet
    OUT.mkdir(parents=True, exist_ok=True); res = dict(node='E37', arm='verify', seeds=dict(fit=VF, test=VT), per_class=PER)
    save = lambda: (OUT / 'e37_verify.json').write_text(json.dumps(E.jsonable(res), indent=1))
    xF, yF, _ = sample(PER, VF); xT, yT, _ = sample(PER, VT); rF, rT = E.rows_of(PER, VF), E.rows_of(PER, VT)
    orig = np.r_[E.rows_of(2000, 3701), E.rows_of(2000, 3702)]
    res['overlap'] = dict(F_with_E37=int(np.isin(rF, orig).sum()), T_with_E37=int(np.isin(rT, orig).sum()), T_with_F=int(np.isin(rT, rF).sum()))
    log('overlap removed', res['overlap'])
    DF = E.prep(np.asarray(xF, np.float32), yF, rF, VF, exclude_rows=orig)
    DT = E.prep(np.asarray(xT, np.float32), yT, rT, VT, exclude_rows=np.r_[orig, rF])
    res['N'] = dict(F_WZ=DF['J'], T_WZ=DT['J'], accepted_T={str(dc): int(DT['acc'][dc].sum()) for dc in E.DCS}, cross_T=int(DT['cross'][2].sum()),
                    max_rel_p4_change=max(DF['dP'], DT['dP']))
    log('N', res['N'])
    net = JNet(); net.off = np.zeros(5)
    res['acc_T_all_stored'] = float((net.net.run(np.asarray(xT, np.float32))['logits'].argmax(1) == yT).mean())
    RF = E.reps_all(net, DF); RT = E.reps_all(net, DT); wzm = E.margin(RT['id']['logits'])
    res['acc_T_WZ_rebuilt'] = float((RT['id']['logits'].argmax(1) == DT['y']).mean())
    for kd, dc in (('D', 0.3), ('cn', 0.3), ('D', -0.3)):
        dm = E.margin(RT[(dc, kd)]['logits']) - wzm; res[f'dmargin_{kd}_{dc:+.2f}'] = dict(W=float(dm[DT['y'] == 2].mean()), Z=float(dm[DT['y'] == 3].mean()))
    for f in (0.95, 1.05):
        dm = E.margin(RT[('dil', f)]['logits']) - wzm; res[f'dmargin_dil_{f}'] = dict(W=float(dm[DT['y'] == 2].mean()), Z=float(dm[DT['y'] == 3].mean()))
    log('acc', res['acc_T_all_stored'], res['acc_T_WZ_rebuilt'], {k: v for k, v in res.items() if k.startswith('dmargin')}); save()
    QS = dict(np.load(OUT / 'e37_das_Q.npz')); MS = dict(np.load(OUT / 'e37_das_M.npz'))
    bt = np.random.default_rng(17); W = np.stack([np.bincount(bt.integers(0, DT['J'], DT['J']), minlength=DT['J']) for _ in range(1000)]).astype(np.float32)
    P = {st: (E.build_pairs(DT, RT, st), E.build_pairs(DF, RF, st)) for st in ('S', 'h4', 'h5')}
    # (1) stored subspaces on fresh jets
    res['stored'] = {}
    for st, k in (('h4', 1), ('h4', 2), ('h4', 3), ('h5', 1), ('S', 1)):
        o = ev(net, st, QS[f'{st}_k{k}'], P[st][0], W); res['stored'][f'{st}|{k}'] = {nm: short(m) for nm, m in o.items()}
        o2 = ev(net, st, QS[f'{st}_k{k}'], P[st][0], None, sets=tuple(f'self{dc:+.2f}' for dc in E.DCS))
        res['stored'][f'{st}|{k}']['per_dc_frac'] = {nm: m['frac'] for nm, m in o2.items()}
        log(f'stored {st} k={k}: ' + json.dumps(E.jsonable({nm: (m['frac'], m['iia_wz'], m['slope']) for nm, m in o.items()})))
    live = [int(u) for u in np.flatnonzero(RF['id']['h4'].std(0) > 1e-9)]
    for k in (1, 2):
        N = rand_null(net, 'h4', P['h4'][0], live, 64, k, 300, 900 + k); res['stored'][f'h4|{k}']['random'] = summ_null(N, res['stored'][f'h4|{k}'])
        log(f'random h4 k={k}: ' + json.dumps(E.jsonable(res['stored'][f'h4|{k}']['random'])))
    save()
    # (2) refit at h4 k=1 on F' (self + cross pairs), and on deranged targets
    PF = P['h4'][1]; fit = [PF['self'], PF['cross']]; hb = np.concatenate([p.hb for p in fit]); hs = np.concatenate([p.hs for p in fit])
    tg = np.concatenate([p.mt for p in fit]); grp = np.concatenate([p.jet for p in fit])
    Qn, info = E.das_fit_margin(net, 'h4', hb, hs, tg, grp, 1, live, seed=101, steps=400)
    o = ev(net, 'h4', Qn, P['h4'][0], W); res['refit_h4_k1'] = dict(train=info, test={nm: short(m) for nm, m in o.items()},
                                                                  abs_cos_with_stored=float(abs(Qn[:, 0] @ QS['h4_k1'][:, 0])))
    log('refit h4 k=1: ' + json.dumps(E.jsonable(res['refit_h4_k1'])))
    pr = np.random.default_rng(201); pi = pr.permutation(len(tg))
    Qp, ip = E.das_fit_margin(net, 'h4', hb, hs, tg[pi], grp, 1, live, seed=301, steps=400)
    o = ev(net, 'h4', Qp, P['h4'][0], W); res['shuffled_h4_k1'] = dict(train=ip, test={nm: short(m) for nm, m in o.items()},
                                                                     abs_cos_with_stored=float(abs(Qp[:, 0] @ QS['h4_k1'][:, 0])))
    log('shuffled h4 k=1: ' + json.dumps(E.jsonable(res['shuffled_h4_k1']))); jax.clear_caches(); save()
    # (3) non-learned k=1 baselines at h4 (fit sample only)
    d = PF['self'].hs - PF['self'].hb; lm = np.isin(np.arange(64), live)
    pc1 = np.linalg.svd(d * lm, full_matrices=False)[2][0]
    dcs_self = np.concatenate([np.full(int(DF['acc'][dc].sum()), dc) for dc in E.DCS])
    md = d[dcs_self == 0.3].mean(0) - d[dcs_self == -0.3].mean(0)
    H = RF['id']['h4']; mF = E.margin(RF['id']['logits']); A = np.c_[np.ones(len(H)), H[:, lm]]; b = np.linalg.lstsq(A, mF, rcond=None)[0]
    gr = np.zeros(64); gr[lm] = b[1:]
    res['baselines_h4_k1'] = {}
    for nm, v in (('pc1_displacement', pc1), ('mean_displacement_dc+0.3_minus_-0.3', md), ('margin_readout_lstsq', gr), ('stored_dilation_direction', MS['dil_h4'][:, 0])):
        v = v * lm; v = (v / np.linalg.norm(v))[:, None]; o = ev(net, 'h4', v, P['h4'][0], W)
        res['baselines_h4_k1'][nm] = dict(test={k_: short(m) for k_, m in o.items()}, abs_cos_with_stored=float(abs(v[:, 0] @ QS['h4_k1'][:, 0])))
        log(f'baseline {nm}: |cos| with DAS {res["baselines_h4_k1"][nm]["abs_cos_with_stored"]:.3f}; ' + json.dumps(E.jsonable({k_: (m['frac'], m['iia_wz'], m['slope']) for k_, m in o.items()})))
    save()
    # (4) positive control: the stored M2 / M0 subspaces at h4 (k = 2)
    res['mass_subspaces_h4'] = {}
    for V in ('M2', 'M0'):
        M = MS[f'{V}_h4'].copy(); M[~lm] = 0; M = np.linalg.qr(M)[0]; o = ev(net, 'h4', M, P['h4'][0], W)
        res['mass_subspaces_h4'][V] = {k_: short(m) for k_, m in o.items()}
        log(f'{V} subspace h4: ' + json.dumps(E.jsonable({k_: (m['frac'], m['iia_wz'], m['slope']) for k_, m in o.items()})))
    save()
    # (5) steering along the k=1 direction and the k=2 read-out direction (E37's), 100 random directions each
    dm_dir = MS['dil_h4'][:, 0]; res['steering'] = {}
    PFs = PF['self']
    for nm, Q in (('h4_k1', QS['h4_k1']), ('h4_k2_readout', QS['h4_k2'])):
        dQ = (PFs.hs - PFs.hb) @ Q; w = np.linalg.lstsq(dQ, PFs.mt - PFs.mb, rcond=None)[0]; v = Q @ w; v = v / np.linalg.norm(v)
        u0 = RT['id']['h4'] @ v
        dU = {f'{dc:+.2f}': {c: float((RT[(dc, 'D')]['h4'][DT['y'] == ci] @ v - u0[DT['y'] == ci]).mean()) for ci, c in ((2, 'W'), (3, 'Z'))} for dc in E.DCS}
        s = E.steering(net, DT, RT, v, 'h4', 100, dm_dir); s['delta_u_per_dc'] = dU; res['steering'][nm] = s
        log(f'steering {nm}: ' + json.dumps(E.jsonable(s))); jax.clear_caches(); save()
    res['wall_s'] = time.time() - T0; save(); log('E37 verify DONE')


def part_b():
    """(6) hold version on D(+0.3) self pairs at h4: h' = h_D + QQ^T(h_base - h_D) -> share of the D margin change removed, per class,
    for the DAS k=1 direction, the M2 (k=2) subspace, the dilation direction, and 200 random k=1 / k=2 live subspaces (E35 framing).
    (7) steering null for the class-opposite sign: fixed offset c x 2.87 (mean |du| of D(+0.3) along the k=1 direction) along the
    DAS k=1 direction and along 200 random live directions, c = -1, +1; statistic Z-W contrast of mean margin changes; 1,000 jet
    bootstrap for the DAS direction.  Same fresh jets as main().  Writes e37_verify_b.json."""
    from interp.jedi_core import sample
    from interp.x10_das_jedi import JNet, proj
    res = dict(node='E37', arm='verify_b'); save = lambda: (OUT / 'e37_verify_b.json').write_text(json.dumps(E.jsonable(res), indent=1))
    xT, yT, _ = sample(PER, VT); rT = E.rows_of(PER, VT); rF = E.rows_of(PER, VF)
    orig = np.r_[E.rows_of(2000, 3701), E.rows_of(2000, 3702)]
    DT = E.prep(np.asarray(xT, np.float32), yT, rT, VT, exclude_rows=np.r_[orig, rF])
    net = JNet(); net.off = np.zeros(5); RT = E.reps_all(net, DT); y = DT['y']
    QS = dict(np.load(OUT / 'e37_das_Q.npz')); MS = dict(np.load(OUT / 'e37_das_M.npz'))
    live = [int(u) for u in np.flatnonzero(RT['id']['h4'].std(0) > 1e-9)]; lm = np.isin(np.arange(64), live)
    hb, hd = RT['id']['h4'], RT[(0.3, 'D')]['h4']; mb = E.margin(RT['id']['logits']); md = E.margin(RT[(0.3, 'D')]['logits'])
    def removed(Q):
        mh = E.margin(net.exact('h4', proj(net.rng_['h4'], hd, hd + ((hb - hd) @ Q) @ Q.T))[0])
        return {c: float(1 - (mh[y == ci] - mb[y == ci]).mean() / (md[y == ci] - mb[y == ci]).mean()) for ci, c in ((2, 'W'), (3, 'Z'))}
    M2 = MS['M2_h4'].copy(); M2[~lm] = 0; M2 = np.linalg.qr(M2)[0]; dl = MS['dil_h4'].copy(); dl[~lm] = 0; dl /= np.linalg.norm(dl)
    res['hold_D03'] = {nm: removed(Q) for nm, Q in (('das_k1', QS['h4_k1']), ('das_k2', QS['h4_k2']), ('M2_k2', M2), ('dilation_dir', dl))}
    rr = np.random.default_rng(31)
    for k in (1, 2):
        R = [removed(E.rand_live(rr, live, 64, k)) for _ in range(200)]
        res['hold_D03'][f'random_k{k}'] = {c: dict(mean=float(np.mean([r[c] for r in R])), q=np.quantile([r[c] for r in R], [.025, .5, .95, .975]).tolist()) for c in ('W', 'Z')}
    log('hold: ' + json.dumps(E.jsonable(res['hold_D03']))); save()
    # dilation pairs: hold of the dilation x0.95 / x1.05 response by the same subspaces (positive control of the hold)
    res['hold_dil'] = {}
    for f in (0.95, 1.05):
        hf = RT[('dil', f)]['h4']; mf = E.margin(RT[('dil', f)]['logits']); o = {}
        for nm, Q in (('das_k1', QS['h4_k1']), ('M2_k2', M2), ('dilation_dir', dl)):
            mh = E.margin(net.exact('h4', proj(net.rng_['h4'], hf, hf + ((hb - hf) @ Q) @ Q.T))[0])
            o[nm] = {c: float(1 - (mh[y == ci] - mb[y == ci]).mean() / (mf[y == ci] - mb[y == ci]).mean()) for ci, c in ((2, 'W'), (3, 'Z'))}
        res['hold_dil'][str(f)] = o
    log('hold dil: ' + json.dumps(E.jsonable(res['hold_dil']))); save()
    v = QS['h4_k1'][:, 0]; u0 = hb @ v; scale = float(np.abs(hd @ v - u0).mean()); sgn = float(np.sign((hd @ v - u0).mean()))
    def steer(g, c):
        lo = net.exact('h4', proj(net.rng_['h4'], hb, hb + c * scale * g[None, :]))[0]; return E.margin(lo) - mb
    res['steer'] = dict(scale=scale, d03_direction_sign=sgn, D03=dict(W=float((md - mb)[y == 2].mean()), Z=float((md - mb)[y == 3].mean())))
    bt = np.random.default_rng(32); BW = [np.bincount(bt.integers(0, len(y), len(y)), minlength=len(y)).astype(float) for _ in range(1000)]
    for c in (-1.0, 1.0):
        dm = steer(v, c * sgn); o = dict(W=float(dm[y == 2].mean()), Z=float(dm[y == 3].mean()))
        ci = {cl: [] for cl in ('W', 'Z', 'ZminusW')}
        for w in BW:
            a = (w * dm)[y == 2].sum() / w[y == 2].sum(); b = (w * dm)[y == 3].sum() / w[y == 3].sum(); ci['W'].append(a); ci['Z'].append(b); ci['ZminusW'].append(b - a)
        o.update({f'{k_}_ci': E.qci(v_) for k_, v_ in ci.items()}); o['ZminusW'] = o['Z'] - o['W']
        res['steer'][f'das_k1_c{c:+.0f}_along_D03' if c > 0 else f'das_k1_c{c:+.0f}_against_D03'] = o
    log('steer das: ' + json.dumps(E.jsonable(res['steer'])))
    rr = np.random.default_rng(33); RS = []
    for _ in range(200):
        g = E.rand_live(rr, live, 64, 1)[:, 0]; row = {}
        for c in (-1.0, 1.0):
            dm = steer(g, c); row[c] = (float(dm[y == 2].mean()), float(dm[y == 3].mean()))
        RS.append(row)
    best = np.array([max(r[c][1] - r[c][0] for c in r) for r in RS]); opp = np.array([any(r[c][0] < 0 < r[c][1] for c in r) for r in RS])
    obs = res['steer']['das_k1_c+1_along_D03']['ZminusW']
    res['steer']['random'] = dict(n=200, ZminusW_max_over_sign_q=np.quantile(best, [.5, .95, .99]).tolist(), frac_dirs_with_W_down_Z_up=float(opp.mean()),
                                  p_ZminusW=float((1 + (best >= obs).sum()) / 201),
                                  W_down_Z_up_mean_given_opp=[float(np.mean([max(r[c][1] - r[c][0] for c in r) for r, o_ in zip(RS, opp) if o_]))] if opp.any() else None)
    log('steer random: ' + json.dumps(E.jsonable(res['steer']['random']))); res['wall_s'] = time.time() - T0; save(); log('E37 verify b DONE')


def part_c():
    """(8) sign-antisymmetric steering test: for the DAS k=1 direction and 200 random live h4 directions, fixed offsets +-c x 2.87
    (c = 0.5, 1) on unedited test jets; A = [(Z-W)(+c) - (Z-W)(-c)] / 2 (the part of the W/Z contrast that flips with the sign of
    the offset; a perturbation that only pushes jets toward the W/Z boundary gives A = 0) and S = [(Z-W)(+c) + (Z-W)(-c)] / 2.
    Random directions: |A| (sign free).  (9) per-dc interchange frac on T for the M2 (k=2) subspace and the dilation direction.
    Same fresh test jets.  Writes e37_verify_c.json."""
    from interp.jedi_core import sample
    from interp.x10_das_jedi import JNet, proj
    res = dict(node='E37', arm='verify_c'); save = lambda: (OUT / 'e37_verify_c.json').write_text(json.dumps(E.jsonable(res), indent=1))
    xT, yT, _ = sample(PER, VT); rT = E.rows_of(PER, VT); rF = E.rows_of(PER, VF)
    orig = np.r_[E.rows_of(2000, 3701), E.rows_of(2000, 3702)]
    DT = E.prep(np.asarray(xT, np.float32), yT, rT, VT, exclude_rows=np.r_[orig, rF])
    net = JNet(); net.off = np.zeros(5); RT = E.reps_all(net, DT); y = DT['y']
    QS = dict(np.load(OUT / 'e37_das_Q.npz')); MS = dict(np.load(OUT / 'e37_das_M.npz'))
    live = [int(u) for u in np.flatnonzero(RT['id']['h4'].std(0) > 1e-9)]; lm = np.isin(np.arange(64), live)
    hb, hd = RT['id']['h4'], RT[(0.3, 'D')]['h4']; mb = E.margin(RT['id']['logits'])
    v = QS['h4_k1'][:, 0]; u0 = hb @ v; scale = float(np.abs(hd @ v - u0).mean()); sgn = float(np.sign((hd @ v - u0).mean())); v = v * sgn
    def zw(g, c):
        dm = E.margin(net.exact('h4', proj(net.rng_['h4'], hb, hb + c * scale * g[None, :]))[0]) - mb
        return float(dm[y == 3].mean() - dm[y == 2].mean()), float(dm[y == 2].mean()), float(dm[y == 3].mean())
    res['scale'] = scale; res['das_k1'] = {}
    for c in (0.5, 1.0):
        p, m = zw(v, c), zw(v, -c); res['das_k1'][str(c)] = dict(plus=p, minus=m, A=(p[0] - m[0]) / 2, S=(p[0] + m[0]) / 2)
    rr = np.random.default_rng(34); RA = {0.5: [], 1.0: []}; RSy = {0.5: [], 1.0: []}
    for _ in range(200):
        g = E.rand_live(rr, live, 64, 1)[:, 0]
        for c in (0.5, 1.0):
            p, m = zw(g, c), zw(g, -c); RA[c].append(abs(p[0] - m[0]) / 2); RSy[c].append((p[0] + m[0]) / 2)
    res['random'] = {str(c): dict(n=200, absA_q=np.quantile(RA[c], [.5, .95, .99]).tolist(), S_q=np.quantile(RSy[c], [.05, .5, .95]).tolist(),
                                  p_absA=float((1 + (np.array(RA[c]) >= abs(res['das_k1'][str(c)]['A'])).sum()) / 201)) for c in (0.5, 1.0)}
    log('antisym steering: ' + json.dumps(E.jsonable(dict(das=res['das_k1'], random=res['random'])))); save()
    PT = E.build_pairs(DT, RT, 'h4')
    M2 = MS['M2_h4'].copy(); M2[~lm] = 0; M2 = np.linalg.qr(M2)[0]; dl = MS['dil_h4'].copy(); dl[~lm] = 0; dl /= np.linalg.norm(dl)
    res['per_dc_frac'] = {}
    for nm, Q in (('das_k1', QS['h4_k1']), ('M2_k2', M2), ('dilation_dir', dl)):
        o = {}
        for dc in E.DCS:
            P_ = PT[f'self{dc:+.2f}']; lp = E.patch(net, 'h4', P_.hb, P_.hs, Q); dp = E.margin(lp) - P_.mb; dt = P_.mt - P_.mb
            o[f'{dc:+.2f}'] = {c: dict(patch=float(dp[y[P_.jet] == ci].mean()), target=float(dt[y[P_.jet] == ci].mean())) for ci, c in ((2, 'W'), (3, 'Z'))}
        res['per_dc_frac'][nm] = o
    log('per dc: ' + json.dumps(E.jsonable(res['per_dc_frac']))); res['wall_s'] = time.time() - T0; save(); log('E37 verify c DONE')


if __name__ == '__main__':
    import sys
    part = sys.argv[-1] if '--part' in sys.argv else 'a'
    dict(a=main, b=part_b, c=part_c)[part]()
