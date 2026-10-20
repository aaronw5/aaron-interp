"""E23w verify: estimator check of the two-gain read-out on a fresh sample.

Question: E23w reports a1/a2 (odd, centroid-linear gain / even, quadratic gain) of 0.4-0.8 per stage with a projection
estimator (regress the translation's odd part O_T on the matched dilation's odd part O_C).  O_C is itself a noisy network
response, so the projection is an errors-in-variables regression and is biased toward 0 when O_C carries response that is
not the predicted log-M2 change; the bias is larger for the odd part (small signal) than for the even part.
Checks run here (all on the same edits as E23w: rigid translations |c| = 10/20/40/60 mrad x 8 azimuths of the 5 %
most off-centre jets, matched dilation by exp(pred), pred = 1/2 dlog(M0 M2)):
  1. reproduction of the E23w projection a1, a2, a1/a2 per stage on sample(5000, SEED) minus the jets of seeds 2311, 2312;
  2. instrumented gains (IV): per unit, O_T and O_C are each regressed on the analytic odd prediction P1 = 1/2 (pred(c) -
     pred(-c)) with the fixed-direction terms c_eta, c_phi; a1_IV = sum_u bT_u bC_u / sum_u bC_u^2 over the stage's units.
     Even: E_T and E_C regressed on P2 = 1/2 (pred(c) + pred(-c)) (variant: plus a jet-independent |c|^2 term).
     A unit that is any function of M0 M2 plus response unrelated to the edit gives a1_IV = a2_IV = 1 in expectation;
  3. positive / analytic controls through both estimators (no network): log(M0 M2) (exact 1 / 1), log M2 quantised in
     steps of 0.02, log M2 plus a pT-weighted half-plane ReLU sum (a non-mass term), and the E-series S-object formulas
     log M2 over ranks >= 1, log M2 over the 14 hardest, sum_{rank>=1} (r - 0.2)_+, sum r, sum r^2;
  4. variance explained (R^2, stage pooled) of the translation's odd part by c alone, by [P1, c], and by alternative
     centres (M1 of the 14 hardest, of ranks >= 1, unweighted centroid, leading particle); same for O_C;
  5. bulk jets (d below the median): total odd RMS of the translation vs its M1-linear (IV) part;
  6. untrained redraws (2): the same estimators.
Jet Poisson bootstrap (NB 1000) for every gain, ratio, and the paired difference IV - projection.
  ./job.sh e23w-verify 'python -m interp.e23w_verify --seed 2313'   -> $AI_RESULTS/jedi/e23w_verify.json
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp.e23w_gains_jedi import moments, shift, scale_ang, cells, jedi_feat, LIVE_S, RINGS, NAZ, holm, ci, pnorm, clean  # noqa: E402

OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
NB = 1000; T0 = time.time()


def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def parts(x):
    x = np.asarray(x, np.float64); rl = x[..., 0] > 0
    pt = np.where(rl, x[..., 0], 0); e = np.where(rl, x[..., 1], 0); p = np.where(rl, x[..., 2], 0)
    rk = np.argsort(np.argsort(-pt, 1), 1)
    return pt, e, p, rl, rk


def analytic_raw(x):
    pt, e, p, rl, rk = parts(x); r2 = e * e + p * p; r = np.sqrt(r2); soft = rl & (rk >= 1); hard = rl & (rk < 14)
    M0 = pt.sum(1); M2 = (pt * r2).sum(1)
    return dict(logM0M2=np.log(M0 * M2), logM2_quant002=np.floor(np.log(M2) / 0.02) * 0.02,
                halfplane=np.where(rl, pt * np.maximum(0.6 * e + 0.8 * p - 0.02, 0), 0).sum(1),
                logM2_rank_ge1=np.log(np.where(soft, pt * r2, 0).sum(1) + 1e-12), logM2_hard14=np.log(np.where(hard, pt * r2, 0).sum(1) + 1e-12),
                S1_like=np.where(soft, np.maximum(r - 0.2, 0), 0).sum(1), sum_r=np.where(rl, r, 0).sum(1), sum_r2=np.where(rl, r2, 0).sum(1))


def make_analytic(x_all):
    R0 = analytic_raw(x_all); s1 = R0['logM0M2'].std(); s2 = R0['halfplane'].std()
    names = ['logM0M2', 'logM2_quant002', 'logM2_plus_halfplane', 'logM2_rank_ge1', 'logM2_hard14', 'S1_like', 'sum_r', 'sum_r2']

    def fn(x):
        R = analytic_raw(x); R['logM2_plus_halfplane'] = R['logM0M2'] / s1 + R['halfplane'] / s2
        return np.stack([R[n] for n in names], 1)
    return fn, {n: [i] for i, n in enumerate(names)}


def centres(x):
    """per-jet odd-direction vectors v (J, 2): the odd prediction of each candidate is c . v"""
    pt, e, p, rl, rk = parts(x); r2 = e * e + p * p; M0 = pt.sum(1); M2 = (pt * r2).sum(1)
    out = {}
    for nm, m in (('M1_full', rl), ('M1_hard14', rl & (rk < 14)), ('M1_rank_ge1', rl & (rk >= 1))):
        w = np.where(m, pt, 0); out[nm] = np.stack([(w * e).sum(1), (w * p).sum(1)], 1) / np.maximum((w * r2).sum(1), 1e-12)[:, None]
    n = rl.sum(1); out['centroid_unweighted'] = np.stack([e.sum(1), p.sum(1)], 1) / np.maximum(np.where(rl, r2, 0).sum(1), 1e-12)[:, None]
    il = np.argmax(pt, 1); jj = np.arange(len(x))
    out['lead'] = np.stack([e[jj, il], p[jj, il]], 1) * (M0 / M2)[:, None]
    del n
    return out


def responses(fn, xs, M, sdU):
    C = cells(); U0 = fn(xs); J = len(xs); K = U0.shape[1]
    DT = np.zeros((len(C), J, K)); DC = np.zeros_like(DT); PRED = np.zeros((len(C), J))
    for i, (r, k, ce, cp) in enumerate(C):
        pred = 0.5 * np.log1p((2 * (ce * M['M1e'] + cp * M['M1p']) + r * r * M['M0']) / M['M2']); PRED[i] = pred
        DT[i] = fn(shift(xs, ce, cp)) - U0; DC[i] = fn(scale_ang(xs, np.exp(pred))) - U0
    I = [i for i, (r, k, _, _) in enumerate(C) if k < NAZ // 2]; Jx = [i + NAZ // 2 for i in I]
    return dict(OT=0.5 * (DT[I] - DT[Jx]) / sdU, ET=0.5 * (DT[I] + DT[Jx]) / sdU, OC=0.5 * (DC[I] - DC[Jx]) / sdU, EC=0.5 * (DC[I] + DC[Jx]) / sdU,
                P1=0.5 * (PRED[I] - PRED[Jx]), P2=0.5 * (PRED[I] + PRED[Jx]), cvec=np.array([[C[i][2], C[i][3]] for i in I]),
                ring=np.array([RINGS.index(C[i][0]) for i in I]), J=J, K=K)


def jetstats(X, Y):
    return np.einsum('pja,pjb->jab', X, X), np.einsum('pja,pju->jua', X, Y), (Y ** 2).sum(0)


def fit(XtX, XtY, w):
    """weighted pooled OLS per unit -> beta (K, a), explained SS (K,); w (J,) or (B, J)"""
    if w.ndim == 1:
        A = np.einsum('j,jab->ab', w, XtX); Bm = np.einsum('j,jua->ua', w, XtY); beta = np.linalg.solve(A, Bm.T).T
        return beta, (beta * Bm).sum(1)
    A = np.einsum('nj,jab->nab', w, XtX); J, K, a = XtY.shape; Bm = (w @ XtY.reshape(J, K * a)).reshape(len(w), K, a)
    beta = np.linalg.solve(A, np.transpose(Bm, (0, 2, 1)))                  # (B, a, K)
    return np.transpose(beta, (0, 2, 1)), None


def odd_design(P1, cvec):
    Pn, J = P1.shape
    return np.concatenate([P1[..., None], np.broadcast_to(cvec[:, None, :], (Pn, J, 2))], -1)


def estimators(D, stages, WB, sel=None):
    """projection (E23w) and IV gains per stage, with bootstrap; sel: boolean over pairs (per ring)"""
    s = np.ones(len(D['cvec']), bool) if sel is None else sel
    OT, OC, ET, EC, P1, P2, cv = (D[k][s] for k in ('OT', 'OC', 'ET', 'EC', 'P1', 'P2', 'cvec')); J = D['J']
    # projection (E23w): O_C residualised on c (pooled, unweighted)
    A = np.einsum('pa,pb->ab', cv, cv) * J; bC = np.linalg.solve(A, np.einsum('pa,pju->au', cv, OC)); OCt = OC - np.einsum('pa,au->pu', cv, bC)[:, None, :]
    n1, d1, n2, d2 = (OT * OCt).sum(0), (OCt ** 2).sum(0), (ET * EC).sum(0), (EC ** 2).sum(0)       # (J, K)
    Xo = odd_design(P1, cv); Xe = P2[..., None]; Xe2 = np.concatenate([P2[..., None], np.broadcast_to((cv ** 2).sum(1)[:, None, None], P2.shape + (1,))], -1)
    S = {nm: jetstats(X, Y) for nm, X, Y in (('oT', Xo, OT), ('oC', Xo, OC), ('eT', Xe, ET), ('eC', Xe, EC), ('e2T', Xe2, ET), ('e2C', Xe2, EC))}
    one = np.ones(J); B = {k: fit(v[0], v[1], one) for k, v in S.items()}; BB = {k: fit(v[0], v[1], WB)[0] for k, v in S.items()}
    out = {}
    for st, idx in stages.items():
        idx = np.asarray(idx)
        def iv(bt, bc): return (bt[..., idx, 0] * bc[..., idx, 0]).sum(-1) / (bc[..., idx, 0] ** 2).sum(-1)
        pa1 = n1[:, idx].sum() / d1[:, idx].sum(); pa2 = n2[:, idx].sum() / d2[:, idx].sum()
        pb1 = (WB @ n1[:, idx].sum(1)) / (WB @ d1[:, idx].sum(1)); pb2 = (WB @ n2[:, idx].sum(1)) / (WB @ d2[:, idx].sum(1))
        i1 = iv(B['oT'][0], B['oC'][0]); i2 = iv(B['eT'][0], B['eC'][0]); i2b = iv(B['e2T'][0], B['e2C'][0])
        ib1 = iv(BB['oT'], BB['oC']); ib2 = iv(BB['eT'], BB['eC']); ib2b = iv(BB['e2T'], BB['e2C'])
        pr, ir, ir2 = pb1 / pb2, ib1 / ib2, ib1 / ib2b
        sd_ir = np.nanstd(ir)
        def r2(k):  # stage-pooled R^2 of Y explained by the design
            return float(B[k][1][idx].sum() / S[k][2][:, idx].sum())
        out[st] = dict(proj=dict(a1=float(pa1), a1_ci=ci(pb1), a2=float(pa2), a2_ci=ci(pb2), ratio=float(pa1 / pa2), ratio_ci=ci(pr)),
                       iv=dict(a1=float(i1), a1_ci=ci(ib1), a2=float(i2), a2_ci=ci(ib2), ratio=float(i1 / i2), ratio_ci=ci(ir),
                               p_ratio1=pnorm((i1 / i2 - 1) / sd_ir) if sd_ir > 0 else None,
                               a2_with_c2=float(i2b), a2_with_c2_ci=ci(ib2b), ratio_with_c2=float(i1 / i2b), ratio_with_c2_ci=ci(ir2)),
                       diff_iv_minus_proj=dict(ratio=float(i1 / i2 - pa1 / pa2), ci=ci(ir - pr), p_boot=float(min(1, 2 * min((ir - pr <= 0).mean(), (ir - pr >= 0).mean())))),
                       r2=dict(oddT_P1c=r2('oT'), oddC_P1c=r2('oC'), evenT_P2=r2('eT'), evenC_P2=r2('eC')),
                       rms=dict(oddT=float(np.sqrt((OT[..., idx] ** 2).mean())), oddC=float(np.sqrt((OC[..., idx] ** 2).mean())),
                                oddT_M1lin_IV=float(np.sqrt(((B['oT'][0][idx, 0] * P1[..., None]) ** 2).mean())),
                                oddT_fixed_dir=float(np.sqrt(((np.einsum('pa,ua->pu', cv, B['oT'][0][idx, 1:])) ** 2).mean())),
                                evenT=float(np.sqrt((ET[..., idx] ** 2).mean())), evenC=float(np.sqrt((EC[..., idx] ** 2).mean()))))
    return out


def centre_r2(D, xs, stages):
    """stage-pooled R^2 of O_T and O_C by [c], [c . v_alt, c] for each candidate centre, and all candidates together"""
    V = centres(xs); cv = D['cvec']; Pn, J = D['P1'].shape; cb = np.broadcast_to(cv[:, None, :], (Pn, J, 2)); one = np.ones(J)
    designs = {'c_only': cb, 'P1_analytic': odd_design(D['P1'], cv)}
    for nm, v in V.items(): designs[nm] = np.concatenate([np.einsum('pa,ja->pj', cv, v)[..., None], cb], -1)
    designs['all_centres'] = np.concatenate([np.einsum('pa,ja->pj', cv, v)[..., None] for v in V.values()] + [cb], -1)
    out = {}
    for Yn in ('OT', 'OC'):
        Y = D[Yn]; YY = (Y ** 2).sum(0)
        for dn, X in designs.items():
            XtX, XtY, _ = jetstats(X, Y); _, ex = fit(XtX, XtY, one)
            for st, idx in stages.items(): out.setdefault(st, {}).setdefault(Yn, {})[dn] = float(ex[idx].sum() / YY[:, idx].sum())
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--per', type=int, default=5000); ap.add_argument('--seed', type=int, default=2313)
    ap.add_argument('--exclude', default='2311,2312'); ap.add_argument('--nbulk', type=int, default=1000); ap.add_argument('--tag', default='')
    a = ap.parse_args()
    from interp.jedi_core import Jedi, sample
    from interp.e23_tail_jedi import sample_idx
    OUT.mkdir(parents=True, exist_ok=True); JS = OUT / f'e23w_verify{"_" + a.tag if a.tag else ""}.json'
    R = dict(plan='module docstring', seed=a.seed, per=a.per, exclude=a.exclude, nb=NB, rings=RINGS); save = lambda: JS.write_text(json.dumps(clean(R)))
    x, y, _ = sample(a.per, a.seed); x = np.ascontiguousarray(x, np.float32)
    ex = [int(v) for v in a.exclude.split(',') if v]; idx0 = sample_idx(a.per, a.seed)
    keep = np.ones(len(x), bool)
    for e_ in ex: keep &= ~np.isin(idx0, sample_idx(a.per, e_))
    x, y = x[keep], y[keep]; R['n_overlap_removed'] = int((~keep).sum()); J = len(x); R['N'] = J
    net = Jedi(64); F, R0 = jedi_feat(net, x, 'jedi', fixed_S=LIVE_S)
    R['acc'] = float((R0['logits'].argmax(1) == y).mean()); R['K'] = F.K
    log(f'N {J} (removed {R["n_overlap_removed"]}); accuracy {R["acc"]:.4f}; K {F.K}')
    sdU = F(x).std(0); sdU[sdU < 1e-9] = 1.0
    M = moments(x); d = M['d']; rng = np.random.default_rng(a.seed + 5)
    tail = np.sort(np.concatenate([np.flatnonzero((y == c) & (d >= np.quantile(d[y == c], 0.95))) for c in range(5)]))
    bulk = np.sort(rng.choice(np.flatnonzero(d < np.median(d)), a.nbulk, replace=False))
    sub = lambda I: {k: v[I] for k, v in M.items()}
    R['sets'] = {nm: dict(n=int(len(I)), d_mrad_p50=float(1e3 * np.median(d[I])), acc=float((R0['logits'][I].argmax(1) == y[I]).mean())) for nm, I in (('tail', tail), ('bulk', bulk))}
    log('sets', R['sets'])
    xt, Mt = x[tail], sub(tail); WB = np.random.default_rng(a.seed + 11).poisson(1.0, (NB, len(tail))).astype(np.float64)
    stages = {st: list(range(sl.start, sl.stop)) for st, sl in F.sl.items()}

    # network, tail
    D = responses(F, xt, Mt, sdU); log('network responses done')
    R['network_tail'] = estimators(D, stages, WB)
    R['network_tail_per_ring'] = {f'r{r}': estimators(D, stages, WB[:200], sel=(D['ring'] == q)) for q, r in enumerate(RINGS)}
    R['network_tail_centre_r2'] = centre_r2(D, xt, stages)
    ps = [R['network_tail'][s]['iv']['p_ratio1'] or 1.0 for s in stages]; R['iv_holm'] = dict(zip(stages, [float(v) for v in holm(ps)]))
    for st, v in R['network_tail'].items():
        log(f'{st}: proj a1 {v["proj"]["a1"]:.3f} a2 {v["proj"]["a2"]:.3f} r {v["proj"]["ratio"]:.3f} {np.round(v["proj"]["ratio_ci"], 3)} | IV a1 {v["iv"]["a1"]:.3f} '
            f'{np.round(v["iv"]["a1_ci"], 3)} a2 {v["iv"]["a2"]:.3f} r {v["iv"]["ratio"]:.3f} {np.round(v["iv"]["ratio_ci"], 3)} (c2 var {v["iv"]["ratio_with_c2"]:.3f}) | '
            f'diff {v["diff_iv_minus_proj"]["ratio"]:.3f} {np.round(v["diff_iv_minus_proj"]["ci"], 3)} | R2 {v["r2"]} | rms {v["rms"]}')
    for st in stages: log(st, 'centre R2', R['network_tail_centre_r2'][st])
    for rr, v in R['network_tail_per_ring'].items(): log(rr, ' | '.join(f'{st}: proj {w["proj"]["ratio"]:.2f} iv {w["iv"]["ratio"]:.2f}' for st, w in v.items()))
    save(); del D

    # analytic controls (same tail jets, same edits)
    fa, sta = make_analytic(x); sda = fa(x).std(0); sda[sda < 1e-12] = 1.0
    Da = responses(fa, xt, Mt, sda); R['analytic_tail'] = estimators(Da, sta, WB); R['analytic_centre_r2'] = centre_r2(Da, xt, sta)
    for st, v in R['analytic_tail'].items():
        log(f'analytic {st}: proj a1 {v["proj"]["a1"]:.3f} a2 {v["proj"]["a2"]:.3f} r {v["proj"]["ratio"]:.3f} {np.round(v["proj"]["ratio_ci"], 3)} | IV a1 {v["iv"]["a1"]:.3f} a2 {v["iv"]["a2"]:.3f} '
            f'r {v["iv"]["ratio"]:.3f} {np.round(v["iv"]["ratio_ci"], 3)} | R2 {v["r2"]}')
    save(); del Da

    # bulk
    xb, Mb = x[bulk], sub(bulk); WBb = np.random.default_rng(a.seed + 12).poisson(1.0, (NB, len(bulk))).astype(np.float64)
    Db = responses(F, xb, Mb, sdU); R['network_bulk'] = estimators(Db, stages, WBb); del Db
    for st, v in R['network_bulk'].items():
        log(f'bulk {st}: rms {v["rms"]} (tail {R["network_tail"][st]["rms"]}); iv a1 {v["iv"]["a1"]:.3f} a2 {v["iv"]["a2"]:.3f}')
    save()

    # untrained redraws
    R['untrained'] = []
    for sd_ in range(0, 80):
        if len(R['untrained']) >= 2: break
        un_ = Jedi(64, untrained=True, seed=sd_)
        try:
            Fu, Ru = jedi_feat(un_, x[:5000], f'untrained{sd_}')
        except Exception as ex_:  # noqa: BLE001
            log('untrained', sd_, repr(ex_)[:200]); continue
        if len(Fu.live['S']) < 2 or any(len(Fu.live[s_]) == 0 for s_ in ('S', 'h4', 'h5', 'z')): continue
        sdu = Fu(x[:5000]).std(0); sdu[sdu < 1e-9] = 1.0
        Du = responses(Fu, xt, Mt, sdu); gu = estimators(Du, {st: list(range(sl.start, sl.stop)) for st, sl in Fu.sl.items()}, WB[:300]); del Du
        R['untrained'].append(dict(seed=sd_, n_live={k: len(v) for k, v in Fu.live.items()}, gains=gu))
        log(f'untrained {sd_} live {R["untrained"][-1]["n_live"]}: ' + ' | '.join(f'{st}: proj {v["proj"]["ratio"]:.2f} (a2 {v["proj"]["a2"]:.2f}) iv {v["iv"]["ratio"]:.2f} (a1 {v["iv"]["a1"]:.2f} a2 {v["iv"]["a2"]:.2f})' for st, v in gu.items()))
        save()
    R['wall_s'] = time.time() - T0; save(); log('done ->', JS)


if __name__ == '__main__':
    main()
