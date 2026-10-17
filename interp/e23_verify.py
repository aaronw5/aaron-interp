"""E23 verification (JEDI-linear 3-feature n64): rerun of arm (B) on a fresh sample plus controls missing from E23.

Fresh sample: interp.jedi_core.sample(per_class=PER, seed=211) (PER = 20000 -> 100,000 jets); read-out ridge of log m on z
fitted on sample(PER/5, seed=213) minus the jets shared with the main sample (E17 / E23 procedure, helpers imported from
interp.e23_tail_jedi). Sets Tq1, Tq5, Td5, Tq5c (as E23). Edits (real particles only):
  recentre / reflect / rotate / cal_recentre / cal_reflect / recentre_Pfreeze      as E23
  perp       rigid shift by d = |M1|/M0 perpendicular to M1, both signs (+, -) run and averaged per jet: every particle moves by
             exactly the recentre displacement; M1 . shift = 0, so M2 -> M2 + M0 d^2 (pred = 1/2 log(1 + q) > 0); m unchanged.
             H1 (read-out follows sqrt(M0 M2) about the stored axis): dread = gain * pred > 0; H0 (m): 0; generic displacement
             lowering the read-out (the recentre excess as an even term): dread < 0.
  cal_perp   matched real change for perp (offsets * exp(pred_perp)).
  cal4_recentre   offsets * exp(4 pred_recentre), response / 4: linearity / input-quantizer check of the matched real change.
  cal_recentre_Pfreeze   matched real change with the nine angle-only P units frozen (E23 compared the P-frozen recentre with
             the unfrozen matched change).
Per-jet even / odd parts of recentre and reflect with bootstrap CIs; even part compared with the perp response.
Calls: recentre vs perp (displacement-matched) with paired bootstrap; Z->W / W->Z with exact McNemar.
(C) 20,000 bulk jets, random-direction shift at the p50 / p90 / p99 of d (as E23).
  ./job.sh e23-verify 'python -m interp.e23_verify'   -> $AI_RESULTS/jedi/e23_verify.json
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp.e23_tail_jedi import (moments, edit_shift, edit_scale, edit_rot, Boot, Ridge, r2, sample_idx, signflip_p, mcnemar,
                                  holm, quant)

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'
CL = ['g', 'q', 'W', 'Z', 't']; W, Z, T = 2, 3, 4
SEED, SEED_RO = 211, 213
log = lambda *a: print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


def main(PER):
    from interp.jedi_core import Jedi, sample
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    res = dict(per_class=PER, seed=SEED, seed_readout=SEED_RO)
    x, y, _ = sample(per_class=PER, seed=SEED); x = np.ascontiguousarray(x, np.float32); J = len(x); res['N'] = J
    idxA = sample_idx(PER, SEED)
    net = Jedi(64)
    R0 = net.run(x, keep=('z', 'logits', 'P')); z0, L0, P0 = R0['z'], R0['logits'], R0['P']; c0 = L0.argmax(1)
    res['baseline_acc'] = float((c0 == y).mean()); log(f'N {J}, acc {res["baseline_acc"]:.4f} ({time.time() - t0:.0f} s)')

    PR = max(PER // 5, 100)
    xr, yr, _ = sample(per_class=PR, seed=SEED_RO); idxR = sample_idx(PR, SEED_RO); keep = ~np.isin(idxR, idxA)
    xr = np.ascontiguousarray(xr[keep], np.float32); zr = net.run(xr, keep=('z',))['z']; Mr = moments(xr)
    perm = np.random.default_rng(SEED_RO).permutation(len(xr)); fh, th = np.sort(perm[:len(xr) // 2]), np.sort(perm[len(xr) // 2:])
    ridge = Ridge(zr, Mr['logm'], fh[: len(fh) * 3 // 4], fh[len(fh) * 3 // 4:], fh)
    M = moments(x); lm0 = ridge(z0)
    res['readout'] = dict(n_readout=int(keep.sum()), n_overlap_removed=int((~keep).sum()), n_fit=int(len(fh)), alpha=float(ridge.alpha),
                          r2_heldout=r2(ridge(zr[th]), Mr['logm'][th]), r2_main_logm=r2(lm0, M['logm']))
    del xr, zr
    log(f'read-out {res["readout"]}')

    q, d = M['q'], M['d']
    res['moments'] = dict(q=quant(q), d=quant(d))
    sets = dict(Tq1=np.flatnonzero(q >= np.quantile(q, 0.99)), Tq5=np.flatnonzero(q >= np.quantile(q, 0.95)),
                Td5=np.flatnonzero(d >= np.quantile(d, 0.95)),
                Tq5c=np.sort(np.concatenate([np.flatnonzero((y == c) & (q >= np.quantile(q[y == c], 0.95))) for c in range(5)])))
    bulk = np.flatnonzero(q < np.median(q))
    res['sets'] = {k: dict(n=int(len(v)), n_class=[int((y[v] == c).sum()) for c in range(5)], q_p50=float(np.median(q[v])),
                           d_p50=float(np.median(d[v]))) for k, v in sets.items()}
    log(f'sets {res["sets"]}')

    U = np.unique(np.concatenate(list(sets.values()))); pos = -np.ones(J, int); pos[U] = np.arange(len(U))
    xU = x[U]; MU = {k: v[U] for k, v in M.items()}
    ce, cp = MU['M1e'] / MU['M0'], MU['M1p'] / MU['M0']
    X = dict(recentre=edit_shift(xU, -ce, -cp), reflect=edit_shift(xU, ce, cp), perp_p=edit_shift(xU, -cp, ce), perp_m=edit_shift(xU, cp, -ce))
    th_ = 2 * np.arcsin(np.clip(MU['d'] / (2 * np.maximum(MU['rmean'], 1e-9)), 0, 1)); X['rotate'] = edit_rot(xU, th_)
    ME = {k: moments(v) for k, v in X.items()}; pred = {k: ME[k]['sq'] - MU['sq'] for k in X}
    pred_perp = 0.5 * (pred['perp_p'] + pred['perp_m'])
    for nm, p_ in (('cal_recentre', pred['recentre']), ('cal_reflect', pred['reflect']), ('cal_perp', pred_perp), ('cal4_recentre', 4 * pred['recentre'])):
        X[nm] = edit_scale(xU, np.exp(p_)); ME[nm] = moments(X[nm]); pred[nm] = ME[nm]['sq'] - MU['sq']
    RE = {k: net.run(v, keep=('z', 'logits')) for k, v in X.items()}
    K1 = net.KB['l1'][0]; ANG = [int(u) for u in np.flatnonzero((np.abs(K1[0]) < 1e-9) & (np.abs(K1[1:]).sum(0) > 1e-9))]
    maskP = np.zeros(64); maskP[ANG] = 1
    RE['recentre_Pfreeze'] = net.run_jetwise(X['recentre'], {'P': (maskP, P0[U])}, keep=('z', 'logits'))
    RE['cal_recentre_Pfreeze'] = net.run_jetwise(X['cal_recentre'], {'P': (maskP, P0[U])}, keep=('z', 'logits'))
    chk = net.run_jetwise(xU, {'P': (maskP, P0[U])}, keep=('logits',))['logits']
    real = xU[..., 0] > 0
    res['checks'] = dict(angle_only_P_units=ANG, Pfreeze_identity_max_dlogit=float(np.abs(chk - L0[U]).max()),
                         perp_max_abs_dlogm=float(max(np.max(np.abs(ME[k]['logm'] - MU['logm'])[MU['m'] > 1]) for k in ('perp_p', 'perp_m'))),
                         perp_dM1_dot=float(np.max(np.abs((ME['perp_p']['M1e'] - MU['M1e']) * MU['M1e'] + (ME['perp_p']['M1p'] - MU['M1p']) * MU['M1p']) / MU['M0'] ** 2)),
                         perp_vs_recentre_disp_max_rel=float(np.max(np.abs(np.hypot(X['perp_p'][..., 1].astype(float) - xU[..., 1], X['perp_p'][..., 2].astype(float) - xU[..., 2])
                                                                           - np.hypot(X['recentre'][..., 1].astype(float) - xU[..., 1], X['recentre'][..., 2].astype(float) - xU[..., 2]))[real])),
                         recentre_max_abs_dlogm=float(np.max(np.abs(ME['recentre']['logm'] - MU['logm'])[MU['m'] > 1])),
                         padding_unchanged=bool(all(np.array_equal(X[k][~real], xU[~real]) for k in X)),
                         pt_unchanged=bool(all(np.array_equal(X[k][..., 0], xU[..., 0]) for k in X)))
    log(f'checks {res["checks"]} ({time.time() - t0:.0f} s)')

    cU = c0[U]; lmU = lm0[U]
    dr = {k: ridge(RE[k]['z']) - lmU for k in RE}; cc = {k: RE[k]['logits'].argmax(1) for k in RE}
    dr['perp'] = 0.5 * (dr['perp_p'] + dr['perp_m']); pred['perp'] = pred_perp; dr['cal4_recentre'] = dr['cal4_recentre'] / 4; pred['cal4_recentre'] = pred['cal4_recentre'] / 4
    pred['recentre_Pfreeze'] = pred['recentre']; pred['cal_recentre_Pfreeze'] = pred['cal_recentre']
    res['B'] = {}; lp = []
    for nm, S in sets.items():
        s = pos[S]; n = len(s); bt = Boot(n, SEED + n); e = dict(n=n)
        for k in ('recentre', 'reflect', 'rotate', 'perp', 'perp_p', 'perp_m', 'cal_recentre', 'cal_reflect', 'cal_perp', 'cal4_recentre',
                  'recentre_Pfreeze', 'cal_recentre_Pfreeze'):
            e[k] = dict(pred=bt.mean(pred[k][s]), dread=bt.mean(dr[k][s]))
            if k in ('recentre', 'reflect', 'rotate', 'perp', 'recentre_Pfreeze'):
                e[k]['p_perm'] = signflip_p(dr[k][s], SEED + 5)
        for a_, b_ in (('recentre', 'cal_recentre'), ('reflect', 'cal_reflect'), ('perp', 'cal_perp'), ('recentre_Pfreeze', 'cal_recentre_Pfreeze'),
                       ('cal4_recentre', 'cal_recentre'), ('recentre_Pfreeze', 'cal_recentre')):
            e[f'ratio_{a_}_over_{b_}'] = bt.ratio(dr[a_][s], dr[b_][s])
        for k in ('recentre', 'reflect', 'perp', 'cal_recentre', 'cal_perp'):
            e[f'ratio_{k}_over_pred'] = bt.ratio(dr[k][s], pred[k][s])
        ev, od = 0.5 * (dr['recentre'][s] + dr['reflect'][s]), 0.5 * (dr['reflect'][s] - dr['recentre'][s])
        pev, pod = 0.5 * (pred['recentre'][s] + pred['reflect'][s]), 0.5 * (pred['reflect'][s] - pred['recentre'][s])
        e['even'] = dict(obs=bt.mean(ev), pred=bt.mean(pev), ratio=bt.ratio(ev, pev), minus_perp=bt.mean(ev - dr['perp'][s]),
                         p_perm_minus_perp=signflip_p(ev - dr['perp'][s], SEED + 6))
        e['odd'] = dict(obs=bt.mean(od), pred=bt.mean(pod), ratio=bt.ratio(od, pod))
        # recentre excess over its matched real change vs perp shortfall from its matched real change
        e['excess_recentre_minus_cal'] = bt.mean(dr['recentre'][s] - dr['cal_recentre'][s])
        e['excess_perp_minus_cal'] = bt.mean(dr['perp'][s] - dr['cal_perp'][s])
        cb = cU[s]; yy = y[U][s]
        for k in ('recentre', 'reflect', 'rotate', 'perp_p', 'perp_m', 'cal_recentre'):
            c1 = cc[k][s]
            e[k]['changed'] = bt.mean((c1 != cb).astype(float))
            e[k]['ZW'] = mcnemar(int(((cb == Z) & (c1 == W)).sum()), int(((cb == W) & (c1 == Z)).sum()))
            e[k]['acc'] = [float((cb == yy).mean()), float((c1 == yy).mean())]
        chp = 0.5 * ((cc['perp_p'][s] != cb).astype(float) + (cc['perp_m'][s] != cb).astype(float))
        e['changed_recentre_minus_perp'] = bt.mean((cc['recentre'][s] != cb).astype(float) - chp)
        e['changed_recentre_minus_rotate'] = bt.mean((cc['recentre'][s] != cb).astype(float) - (cc['rotate'][s] != cb).astype(float))
        res['B'][nm] = e
        lp.append((nm, e['recentre']['dread']['log10_p_z'])); lp.append((nm + ':perp', e['perp']['dread']['log10_p_z']))
        f = lambda v: f'{1e3 * v["mean"]:+.2f} [{1e3 * v["ci"][0]:+.2f}, {1e3 * v["ci"][1]:+.2f}]'
        log(f'{nm} n {n}: recentre pred {1e3 * e["recentre"]["pred"]["mean"]:+.2f} dread {f(e["recentre"]["dread"])} cal {f(e["cal_recentre"]["dread"])} '
            f'cal4/4 {f(e["cal4_recentre"]["dread"])} obs/cal {e["ratio_recentre_over_cal_recentre"]}')
        log(f'  reflect pred {1e3 * e["reflect"]["pred"]["mean"]:+.2f} dread {f(e["reflect"]["dread"])} obs/cal {e["ratio_reflect_over_cal_reflect"]}')
        log(f'  perp pred {1e3 * e["perp"]["pred"]["mean"]:+.2f} dread {f(e["perp"]["dread"])} (+ {f(e["perp_p"]["dread"])}, - {f(e["perp_m"]["dread"])}) cal {f(e["cal_perp"]["dread"])} '
            f'obs/cal {e["ratio_perp_over_cal_perp"]}')
        log(f'  rotate {f(e["rotate"]["dread"])}; Pfreeze {f(e["recentre_Pfreeze"]["dread"])} cal_Pfreeze {f(e["cal_recentre_Pfreeze"]["dread"])} '
            f'ratio {e["ratio_recentre_Pfreeze_over_cal_recentre_Pfreeze"]}')
        log(f'  even obs {f(e["even"]["obs"])} pred {1e3 * e["even"]["pred"]["mean"]:+.2f} ratio {e["even"]["ratio"]}; even-perp {f(e["even"]["minus_perp"])}; '
            f'odd obs {f(e["odd"]["obs"])} pred {1e3 * e["odd"]["pred"]["mean"]:+.2f} ratio {e["odd"]["ratio"]}')
        log(f'  changed recentre {e["recentre"]["changed"]["mean"]:.4f} perp+ {e["perp_p"]["changed"]["mean"]:.4f} perp- {e["perp_m"]["changed"]["mean"]:.4f} '
            f'rotate {e["rotate"]["changed"]["mean"]:.4f} cal {e["cal_recentre"]["changed"]["mean"]:.4f}; rec-perp {e["changed_recentre_minus_perp"]["mean"]:+.4f} '
            f'{e["changed_recentre_minus_perp"]["ci"]}; ZW rec {e["recentre"]["ZW"]} perp+ {e["perp_p"]["ZW"]}')
    hz = holm([v[1] for v in lp]); res['holm_normal'] = [dict(test=v[0], log10_p=v[1], log10_p_holm=float(h)) for v, h in zip(lp, hz)]

    rngC = np.random.default_rng(SEED + 2); Cj = np.sort(rngC.choice(bulk, min(20000, len(bulk)), replace=False))
    psi = rngC.uniform(0, 2 * np.pi, len(Cj)); xC = x[Cj]; MC = {k: v[Cj] for k, v in M.items()}; btC = Boot(len(Cj), SEED + 23); res['C'] = {}
    for nm, f_ in (('d_p50', 0.5), ('d_p90', 0.9), ('d_p99', 0.99)):
        s_ = float(np.quantile(d, f_)); xe = edit_shift(xC, s_ * np.cos(psi), s_ * np.sin(psi)); Me = moments(xe); Rr = net.run(xe, keep=('z', 'logits'))
        cE = Rr['logits'].argmax(1); cb = c0[Cj]; pr = Me['sq'] - MC['sq']; drC = ridge(Rr['z']) - lm0[Cj]
        res['C'][nm] = dict(size=s_, pred=btC.mean(pr), dread=btC.mean(drC), ratio=btC.ratio(drC, pr), changed=btC.mean((cE != cb).astype(float)),
                            ZW=mcnemar(int(((cb == Z) & (cE == W)).sum()), int(((cb == W) & (cE == Z)).sum())))
        log(f'C {nm} {s_ * 1e3:.2f} mrad: ratio {res["C"][nm]["ratio"]} changed {res["C"][nm]["changed"]["mean"]:.4f} {res["C"][nm]["changed"]["ci"]} ZW {res["C"][nm]["ZW"]}')
    json.dump(res, open(OUT / 'e23_verify.json', 'w'), indent=1, default=float)
    log(f'wrote {OUT / "e23_verify.json"}; DONE ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000)
