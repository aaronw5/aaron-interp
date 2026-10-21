"""E26w verify (skeptic; claims A14 / A3): fresh-sample replicate of E26w's key numbers plus the missing S0-free control.

Sample: interp.jedi_core.sample(per_class=2000, seed=2611) = 10,000 test jets (evaluation); null geometry (covariance Sigma of S,
boost-displacement second moment C_D = E[D D^T], D = S(boosted) - S(unboosted), principal directions) fit on sample(2000, seed 2612).
Edit B (imported from interp.h12_boost_panel): exact Lorentz boost along the jet momentum, |P_jet| -> lam |P_jet|, jet mass fixed.
Hold: network sees the boosted jet, S_hold = S_B + (S_0 - S_B) Q Q^T clipped to the representable range, exact fixed-point tail.
Metric: removed share = 1 - #changed calls with the hold / #changed calls under B alone (as E26w, via interp.e26_boost_route.Ctx).

Replicated (both lam 0.9 / 1.1): holds S0, PC1, S0+S29, DAS M0 (1-dim), DAS M0+M2 (2-dim); iso / variance-matched /
boost-shift-matched live nulls (500 draws); path patch a15 -> g49 -> S0; single live h4 / h5 unit holds vs 2,000 iso live
1-dim directions (Holm per layer).
New controls:
 (a) shift-matched null draws record cos^2 with the S0 axis e0; removed share regressed on cos^2 and reported for draws with
     cos^2 < 0.1 / 0.25 / >= 0.5 (does the shift-matched null remove ~60 % only because its draws contain S0?);
 (b) shift-matched and variance-matched nulls drawn on the live S coordinates WITHOUT S0 (directions orthogonal to e0), same
     targets (S0's boost-displacement second moment / variance); if the target cannot be reached, the failure count and the largest
     reachable value (top eigenvalue of C_D restricted to live minus S0) are reported;
 (c) every live S unit held alone at each lam, with its boost-displacement second moment C_D[u,u], to see where S0 sits
     relative to other units of comparable displacement;
 (d) data check on the n64 test set (260k): stratified AUC of sum pT (14 hardest, all) within 1 GeV and 5 GeV mass bins, with the
     within-bin mass AUC as the residual-mass reference (does the W vs Z number shrink when the bin is narrower?).
Run: ./job.sh e26w-verify 'python -m interp.e26w_verify'   (--smoke N)
Writes $AI_RESULTS/jedi/e26w_verify[_smoke].json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'
T0_ = time.time()
LAMS = (0.9, 1.1); CL = ['g', 'q', 'W', 'Z', 't']


def log(*a):
    print(f'[{time.time() - T0_:7.0f}s]', *a, flush=True)


from interp.e26_boost_route import Net, Ctx, pair_inputs, qci, prand  # noqa: E402
from interp.h12_boost_panel import boost, hls_prep, sortT, mcnemar  # noqa: E402
from interp import x10_das_jedi as X  # noqa: E402
from interp.e26w_boost_specificity import Tilt, iso, capt, StratAUC, jet_feats  # noqa: E402


def data_check(sm, NB):
    from jetdistill.config import RESULTS
    d = np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r'); ya = np.asarray(d['y'])
    J = len(ya) if not sm else 20000; F = {k: np.empty(J) for k in ('m', 'sumpt', 'sumpt14', 'n')}
    for i in range(0, J, 20000):
        f = jet_feats(np.asarray(d['x'][i:min(J, i + 20000)], np.float32))
        for k in F: F[k][i:i + len(f['m'])] = f[k]
    y = ya[:J].astype(int); rng = np.random.default_rng(2611); out = dict(N=int(J))
    for bw in (1.0, 5.0):
        edges = np.arange(0, 300 + bw, bw); b = np.clip(np.digitize(F['m'], edges) - 1, 0, len(edges) - 1); R = out[f'bin_{bw:g}GeV'] = {}
        for ca, cb in ((2, 3), (2, 4), (3, 4), (1, 0)):
            sel = (y == ca) | (y == cb); lab = (y[sel] == ca).astype(np.float64); bb = b[sel]
            cA = np.bincount(bb, lab, len(edges)); cB = np.bincount(bb, 1 - lab, len(edges)); okb = (cA >= 20) & (cB >= 20)
            keep = okb[bb]; lab, bb = lab[keep], bb[keep]; P = dict(n_jets=int(keep.sum()), n_bins=int(okb.sum()))
            for feat in ('sumpt14', 'sumpt', 'm'):
                v = F[feat][sel][keep]; A = StratAUC(v, bb); a0 = A(lab, 1 - lab)
                W = rng.poisson(1.0, (NB, len(v))).astype(np.float64); bt = np.array([A(w * lab, w * (1 - lab)) for w in W])
                P[feat] = dict(auc=a0, ci=qci(bt))
            R[f'{CL[ca]}_vs_{CL[cb]}'] = P
        log(f'data bin {bw} GeV: ' + '; '.join(f'{k} s14 {v["sumpt14"]["auc"]:.3f} s {v["sumpt"]["auc"]:.3f} m {v["m"]["auc"]:.3f}' for k, v in R.items()))
    return out


def main(a):
    from interp.jedi_core import sample
    from interp.h1_irc_panel import hls_table
    sm = a.smoke; pc = sm or 2000; seed = a.seed
    NB = 200 if sm else 1000; NN = 40 if sm else 500; NH = 100 if sm else 2000; NBd = 10 if sm else 100
    OUT.mkdir(parents=True, exist_ok=True); JS = OUT / f'e26w_verify{"_smoke" if sm else ""}.json'
    R = dict(sample=f'interp.jedi_core.sample(per_class={pc}, seed={seed})', fit_sample=f'interp.jedi_core.sample(per_class={pc}, seed={seed + 1})',
             lams=list(LAMS), n_boot=NB, n_null_S=NN, n_null_h=NH)

    def save(): JS.write_text(json.dumps(X.clean(R), indent=1))

    x, y, _ = sample(pc, seed); J = len(y); T0 = hls_table(x); x0 = hls_prep(sortT(T0))
    net = Net(); R0 = net.fwd(x0); c0 = R0['logits'].argmax(1)
    R['N'] = J; R['acc'] = float((c0 == y).mean())
    R['tail_check'] = net.jn.check(x0[:1000], {k: v[:1000] for k, v in R0.items()})
    if not sm:
        R['overlap_with_E26w_eval'] = int(len(np.intersect1d(X.indices(pc, seed), X.indices(pc, 1203))))
        R['overlap_fit_eval'] = int(len(np.intersect1d(X.indices(pc, seed), X.indices(pc, seed + 1))))
    log(f'N {J} acc {R["acc"]:.4f} tail {R["tail_check"]} overlaps {R.get("overlap_with_E26w_eval")} {R.get("overlap_fit_eval")}')
    Wb = np.random.default_rng(seed + 11).poisson(1.0, (NB, J)).astype(np.float32)

    xf, yf, _ = sample(pc, seed + 1); Tf = hls_table(xf); xf0 = hls_prep(sortT(Tf)); Rf0 = net.fwd(xf0, keep=('S', 'h4', 'h5'))
    COV = {st: np.cov(Rf0[st], rowvar=False) for st in ('S', 'h4', 'h5')}; CD = {}
    for lam in LAMS:
        Uf, _ = boost(Tf, lam); RfB = net.fwd(pair_inputs(Tf, Uf)[0], keep=('S',)); D = RfB['S'] - Rf0['S']; CD[lam] = D.T @ D / len(D)
    _, _, Vt = np.linalg.svd(Rf0['S'] - Rf0['S'].mean(0), full_matrices=False)
    del xf, Tf, xf0

    # DAS refit at S (as E26 / E26v / E26w)
    pcF = 300 if sm else 4000; xF, yF, _ = sample(pcF, 0); mF = X.moments(xF); NF = len(yF)
    npair = 800 if sm else 8000; steps = 100 if sm else 800
    r2 = np.random.default_rng(2); fb = r2.integers(0, NF, npair); fs = r2.integers(0, NF, npair)
    while (fb == fs).any(): fs[fb == fs] = r2.integers(0, NF, (fb == fs).sum())
    RF = net.jn.reps(xF, keep=('S', 'logits')); baseF = RF['logits'].argmax(1)[fb]
    labF = {V: net.jn.calls(X.do_edit(xF[fb], V, (mF[0][fb], mF[1][fb]), (mF[0][fs], mF[1][fs]))) for V in ('M0', 'M2')}
    Q = {}
    for V in ('M0', 'M2'):
        Q[V], _ = X.das_fit(net.jn.tail('S'), RF['S'][fb], RF['S'][fs], labF[V], baseF, 1, seed=11, steps=steps, rg=net.jn.rng_['S'])
    Q['M0M2'] = np.linalg.qr(np.concatenate([Q['M0'], Q['M2']], 1))[0]
    del xF, RF
    R['das_M0_cos2_e0'] = float(Q['M0'][0, 0] ** 2)

    RB = {}; XB = {}
    for lam in LAMS:
        U, _ = boost(T0, lam); xB, x0B, o = pair_inputs(T0, U); XB[lam] = (U, xB, x0B); RB[lam] = net.fwd(xB)
    live = {st: np.flatnonzero(np.maximum(R0[st].std(0), np.maximum(*[RB[l][st].std(0) for l in LAMS])) > 1e-9) for st in ('S', 'h4', 'h5')}
    liveS0 = np.setdiff1d(live['S'], [0])
    R['live'] = {k: v.tolist() for k, v in live.items()}
    eye = np.eye(64)
    NAMED = {'S0': eye[:, [0]], 'PC1': Vt[:1].T, 'S0+S29': eye[:, [0, 29]], 'DAS_M0_k1': Q['M0'], 'DAS_M0M2_k1': Q['M0M2']}
    R['geometry'] = {}
    for lam in LAMS:
        C = CD[lam]; e, V = np.linalg.eigh(C[np.ix_(live['S'], live['S'])]); e2 = np.linalg.eigvalsh(C[np.ix_(liveS0, liveS0)])
        dg = np.diag(C); order = live['S'][np.argsort(-dg[live['S']])]
        R['geometry'][str(lam)] = dict(CD_S0=float(C[0, 0]), CD_diag_top=[(int(u), float(dg[u])) for u in order[:8]],
                                       CD_top_eig=e[::-1][:4].tolist(), CD_top_eigvec_cos2_e0=float(V[list(live['S']).index(0), -1] ** 2),
                                       CD_top_eig_without_S0=float(e2.max()), S0_share_of_trCD=float(C[0, 0] / np.trace(C)),
                                       var_S0=float(COV['S'][0, 0]))
    log(f'geometry {R["geometry"]}')

    R['lam'] = {}
    for lam in LAMS:
        t = time.time(); E = R['lam'][str(lam)] = {}; U, xB, x0B = XB[lam]; Rb = RB[lam]
        ctx = Ctx(R0['logits'], Rb['logits'], y, Wb, np.ones(J, bool)); chB = ctx.chB
        E['B'] = dict(changed=float(chB.mean()), ci=qci((Wb @ chB.astype(np.float32)) / Wb.sum(1)), dS0_mean=float((Rb['S'][:, 0] - R0['S'][:, 0]).mean()))
        lo_, hi_ = net.jn.rng_['S']; lo2, hi2 = np.minimum(np.minimum(lo_, Rb['S']), R0['S']), np.maximum(np.maximum(hi_, Rb['S']), R0['S'])
        Ssub = lambda Qm: net.tail('S', np.clip(Rb['S'] + ((R0['S'] - Rb['S']) @ Qm) @ Qm.T, lo2, hi2))[0]
        H = E['holds'] = {}; CH = {}
        for nm, q in NAMED.items():
            LH = Ssub(q); H[nm] = ctx.stats(LH); CH[nm] = LH.argmax(1) != c0
            H[nm]['captured_var'] = capt(COV['S'], q); H[nm]['boost_shift_ms'] = capt(CD[lam], q)
            for kk in ('flow', 'in_window', 'dWQ'): H[nm].pop(kk, None)
        ra = 1 - (Wb @ CH['S0'].astype(np.float32)) / np.maximum(ctx.bB, 1); rb = 1 - (Wb @ CH['PC1'].astype(np.float32)) / np.maximum(ctx.bB, 1)
        E['S0_vs_PC1'] = dict(diff=H['S0']['removed'] - H['PC1']['removed'], ci=qci(ra - rb), mcnemar=mcnemar(CH['S0'], CH['PC1']))
        log(f'lam {lam}: B {E["B"]}; holds ' + '; '.join(f'{k} {v["removed"]:.3f} {np.round(v["removed_ci"], 3).tolist()}' for k, v in H.items()))
        # (c) every live S unit alone
        E['S_units'] = {str(int(u)): dict(removed=float(ctx.rem(Ssub(eye[:, [u]]))), CD=float(CD[lam][u, u]), var=float(COV['S'][u, u])) for u in live['S']}
        log(f'lam {lam} S units: ' + '; '.join(f'{u} {v["removed"]:.3f} CD {v["CD"]:.2f}' for u, v in sorted(E['S_units'].items(), key=lambda kv: -kv[1]['CD'])))
        # nulls
        rq = np.random.default_rng(seed + 100 + int(lam * 10)); NU = E['nulls'] = {}
        v = [ctx.rem(Ssub(iso(live['S'], 64, 1, rq))) for _ in range(NN)]; NU['iso_k1'] = dict(n=len(v), q=np.quantile(v, [.025, .5, .95, .975, 1]).tolist())
        H['S0']['p_iso'] = prand(v, H['S0']['removed'])
        for kind, C_, lv in (('var', COV['S'], live['S']), ('shift', CD[lam], live['S']), ('var_noS0', COV['S'], liveS0), ('shift_noS0', CD[lam], liveS0)):
            Tl = Tilt(C_, lv, 64); tg = C_[0, 0]; vals, c2, fail = [], [], 0
            while len(vals) < NN and fail < 3 * NN:
                Qn, p = Tl.draw(1, tg, rq)
                if Qn is None: fail += 1; continue
                vals.append(ctx.rem(Ssub(Qn))); c2.append(float(Qn[0, 0] ** 2))
            vals, c2 = np.asarray(vals), np.asarray(c2); r = dict(target=float(tg), n=len(vals), failed=fail)
            if len(vals):
                r.update(q=np.quantile(vals, [.025, .5, .95, .975, 1]).tolist(), mean=float(vals.mean()), p_S0=prand(vals, H['S0']['removed']),
                         cos2_e0_q=np.quantile(c2, [.05, .5, .95]).tolist())
                if kind == 'shift':
                    sl = np.polyfit(c2, vals, 1) if c2.std() > 0 else [np.nan, np.nan]
                    r['fit_removed_vs_cos2'] = dict(slope=float(sl[0]), intercept=float(sl[1]), corr=float(np.corrcoef(c2, vals)[0, 1]))
                    for lo_c, hi_c in ((0, .1), (0, .25), (.25, .5), (.5, 1.01)):
                        s = (c2 >= lo_c) & (c2 < hi_c)
                        r[f'cos2_{lo_c}_{hi_c}'] = dict(n=int(s.sum()), median=float(np.median(vals[s])) if s.any() else None,
                                                         q95=float(np.quantile(vals[s], .95)) if s.sum() > 5 else None)
            if kind.endswith('noS0'):
                r['max_reachable'] = float(np.linalg.eigvalsh(C_[np.ix_(lv, lv)]).max())
            NU[kind] = r
        log(f'lam {lam} nulls: ' + json.dumps(X.clean(NU)) + f' p_iso {H["S0"]["p_iso"]} ({time.time() - t:.0f} s)')
        save()

        # single h4 / h5 unit holds vs iso live nulls
        E['h_units'] = {}
        for st in ('h4', 'h5'):
            lo4, hi4 = net.jn.rng_[st]; lb, hb = np.minimum(np.minimum(lo4, Rb[st]), R0[st]), np.maximum(np.maximum(hi4, Rb[st]), R0[st])
            d = Rb[st].shape[1]; hsub = lambda Qm: net.tail(st, np.clip(Rb[st] + ((R0[st] - Rb[st]) @ Qm) @ Qm.T, lb, hb))[0]
            obs = {}
            for u in live[st]:
                h = Rb[st].copy(); h[:, u] = R0[st][:, u]; obs[int(u)] = ctx.rem(net.tail(st, h)[0])
            rh = np.random.default_rng(seed + 500 + int(lam * 10) + (st == 'h5')); vn = np.array([ctx.rem(hsub(iso(live[st], d, 1, rh))) for _ in range(NH)])
            ps = {u: prand(vn, r) for u, r in obs.items()}; hp = X.holm(list(ps.values()))
            named = [u for u in ((53, 39, 3) if st == 'h4' else (24, 25, 1)) if u in obs]; top = sorted(obs, key=lambda u: -obs[u])[:4]
            E['h_units'][st] = dict(n_live=len(obs), sel={str(u): dict(removed=obs[u], holm=float(hp[list(ps).index(u)])) for u in sorted(set(named) | set(top))},
                                    passing=[int(u) for u, h_ in zip(ps, hp) if h_ < 0.05], null_q=np.quantile(vn, [.5, .975, 1]).tolist())
            log(f'lam {lam} {st}: {E["h_units"][st]}')
        save()

        # path patch a15 -> g49 -> S0
        P_ = E['path'] = {}; m49 = np.zeros(64); m49[49] = 1; m15 = np.zeros(64); m15[15] = 1
        Rg0 = net.fwd(xB, {'g': (m49, R0['g'])}, keep=('logits', 'S', 'g')); Rx0gB = net.fwd(x0, {'g': (m49, Rb['g'])}, keep=('logits', 'S'))
        S0_0, S0_B, S0_dir, S0_ctx = R0['S'][:, 0], Rb['S'][:, 0], Rg0['S'][:, 0], Rx0gB['S'][:, 0]

        def setS0(v): h = Rb['S'].copy(); h[:, 0] = v; return net.tail('S', h)[0]
        def st_(L): s = ctx.stats(L); return dict(removed=s['removed'], ci=s['removed_ci'])
        P_['S0_full'] = st_(setS0(S0_0)); P_['S0_context_route_reverted'] = st_(setS0(S0_dir)); P_['S0_direct_route_reverted'] = st_(setS0(S0_ctx))
        P_['g49_all_routes'] = st_(Rg0['logits'])
        h = Rg0['S'].copy(); h[:, 0] = S0_B; Lk = net.tail('S', h)[0]; P_['g49_with_S0_kept_boosted'] = st_(Lk)
        P_['g49_with_S0_kept_boosted_calls_differ_from_B'] = int((Lk.argmax(1) != Rb['logits'].argmax(1)).sum())
        P_['g49_other_S_max_abs_change'] = float(np.abs(np.delete(Rg0['S'] - Rb['S'], 0, 1)).max())
        a0B = net.a_of(x0B); Ra = net.fwd(xB, {'a': (m15, a0B)}, keep=('logits', 'S', 'g')); P_['a15_all_routes'] = st_(Ra['logits'])
        P_['a15_not_via_g'] = st_(net.fwd(xB, {'a': (m15, a0B), 'g': (np.ones(64), Rb['g'])}, keep=('logits',))['logits'])
        del a0B
        dS = S0_B - S0_0
        P_['dS0'] = dict(total=float(dS.mean()), context=float((S0_B - S0_dir).mean()), direct=float((S0_B - S0_ctx).mean()),
                         context_share=float((S0_B - S0_dir).mean() / dS.mean()), under_a15_hold=float((Ra['S'][:, 0] - S0_0).mean()))
        log(f'lam {lam} path: {P_}')
        save()

    if not a.skip_data:
        R['data'] = data_check(sm, NBd); save()
    log(f'wrote {JS}; E26w verify DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=2611); ap.add_argument('--smoke', type=int, default=0)
    ap.add_argument('--skip-data', action='store_true')
    main(ap.parse_args())
