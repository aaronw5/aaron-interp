"""E30 (claims A3, A5, A11; PAPERS.md 'What to look for': weak activations and negative poles): do S29, S1, S0, z0, z7,
z10 have one job across their activation range, and what does the off (negative) side of z0, z7, z10 carry?

Network: JEDI-linear 3-feature n64 (fixed point, interp.jedi_core). Jets: fit = sample(2000, 3001) (10,000), test =
sample(1600, 3002) (8,000) minus jets also in the fit sample.
Units (E5/E5v/E6/E7):
  S29  Σ_slots ReLU(4.44 ln pT + 93.8 r − 21.2)/16 (hinge sum, all 64 slots; correlational proxy ΣpT·r² + Σr² + N(pT<2, r>0.1))
  S1   Σ_{slots ≥ 1} ReLU(0.55 ln pT + 89.3 r − 22.9)/16  (≈ Σ_{ranks ≥ 1}(r − 0.2)₊)
  S0   Σ_{slots 0–13} (0.32 pT + 1.40)/16 directly, plus the jet-context route through g49 (≈ ΣpT of the 14 hardest)
  z0, z7, z10  pre-ReLU d6 neurons; mass windows near 75, 94–98, 135–148 GeV (E6)
Part 1 (weak range): unit value split into deciles (fit-sample quantiles; S units over all jets; z units over jets with z > 0).
  Main formula fitted on the top 3 deciles of the fit jets only; per test decile: R², RMSE/sd, mean residual of that
  top-fit, vs the same form refit on that decile ('same-form') and a free greedy 3-term (S) / 8-term (z) refit ('free').
  S candidates: hand-built energy-flow quantities (all slots and in the unit's slot window, 'K:') + the E5 hinge sum.
  z main formula: E7's K = 32 hinge formula on the 19 live S units (e7_S.json, fitted on 20,000 jets), applied unchanged and with
  its terms refit on the top 3 deciles; free refit: hinge basis S_c, (S_c − t)₊, (t − S_c)₊, 9 quantile knots, greedy 8 terms.
Part 2 (causal dose): per jet one edit; edit slope = Σ measured ΔS / Σ predicted ΔS per decile of the unedited unit value.
  S29, S1  one particle moved radially by Δr ∈ ±{0.01, 0.02, 0.03} at fixed pT and azimuth, chosen among particles in
           the unit's slots whose hinge is on before or after (predicted ΔS = Δhinge/16, E5 coefficients);
           control: same Δr on a particle with predicted ΔS = 0 (S1: slot 0, outside its window; S29 [window = all
           slots]: a particle whose hinge is off before and after).
  S0       one of the 14 hardest particles pT × (1 ± 0.02), ranks re-sorted; predicted ΔS = top-fit coefficient ×
           Δ(ΣpT of the 14 hardest); control: same nudge on a particle of rank ≥ 14 (predicted 0).
  z        S-level intervention: the S unit that carries the largest share of the top-fit formula set to S ± 0.5 sd
           (central difference); predicted Δz from the E7 formula; control: same δ on the live S unit absent from the
           formula with the smallest predicted effect. Untrained control: the same edits on Jedi(untrained=True) (unit measured / trained prediction).
Part 3 (negative poles of z0, z7, z10): class mix and mass of the most negative jets, Spearman with each live S, free
  8-term hinge fit of the negative side (R² on test z < 0); ReLU bypass logits' = logits + α · min(z_k, 0) · W7[k, :]
  (α = 1: the d7 slope of the positive side, the linear read; α = 0.5) vs the trained clamp at 0; McNemar on correctness;
  control: 20 random directions v in h5, z_r = a(v·h5) + b scaled to z_k's negative fraction and sd, passed through the same W7 row.
Statistics: 1,000 bootstrap resamples over jets; H0 slope(decile d) = slope(top 3 deciles): bootstrap and permutation p (2,000
  label shuffles); BH over the 6 units × 7 lower deciles (Part 2) and over the decile RMSE gaps top-fit − free (Part 1).
Output: $AI_RESULTS/jedi/e30_weak_poles.json, e30_weak_poles.npz (figure data), e30_weak_poles.png.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.jedi_core import Jedi, sample
from interp.e4_e5_jedi import handbuilt, hinge_sum, hinge_eval, masks
from interp.e5_causal_jedi import in_range, resort, move_radial

OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'jedi'; OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
LIVE = [0, 1, 13, 17, 19, 20, 29, 30, 31, 40, 43, 50, 51, 52, 56, 58, 60, 62, 63]
SU = [29, 1, 0]; ZU = [0, 7, 10]
NB = 1000; NPERM = 2000
QUICK = '--quick' in sys.argv


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def r2(p, t):
    v = ((t - t.mean()) ** 2).sum()
    return None if v <= 1e-12 else float(1 - ((p - t) ** 2).sum() / v)


def greedy(X, t, k, names):
    """forward selection (column most correlated with the residual), least-squares refit each step"""
    sd = X.std(0); ok = np.flatnonzero(sd > 1e-9)
    Z = (X[:, ok] - X[:, ok].mean(0)) / sd[ok]
    sel = []; res = t - t.mean(); coef = np.array([t.mean()])
    for _ in range(min(k, len(ok))):
        c = np.abs(Z.T @ res); c[sel] = -1; j = int(np.argmax(c))
        if c[j] <= 1e-9: break
        sel.append(j)
        A = np.c_[X[:, ok[sel]], np.ones(len(t))]; coef = np.linalg.lstsq(A, t, rcond=None)[0]; res = t - A @ coef
    cols = [int(ok[s]) for s in sel]
    return dict(cols=cols, terms=[names[c] for c in cols], coef=[float(c) for c in coef[:-1]], b=float(coef[-1]))


def apply(F, X):
    return X[:, F['cols']] @ np.array(F['coef']) + F['b'] if F['cols'] else np.full(len(X), F['b'])


def linfit(X, t):
    A = np.c_[X, np.ones(len(t))]; c = np.linalg.lstsq(A, t, rcond=None)[0]
    return c


def rmse_gap(p_top, p_free, t, sd, rng, nb=300):
    """bootstrap of RMSE(top-fit) − RMSE(free refit) within a decile, in units of the unit's sd"""
    n = len(t); out = []
    for _ in range(nb):
        i = rng.choice(n, n)
        out.append((np.sqrt(((p_top[i] - t[i]) ** 2).mean()) - np.sqrt(((p_free[i] - t[i]) ** 2).mean())) / sd)
    return np.array(out)


def e7_formula(F):
    """E7 hinge formula (results/jedi/e7_S.json, K = 32, fitted on 20,000 jets) -> f(S (J, 64)) and the S units it reads"""
    import re
    ex = []
    for t, c in [(t[0], t[1]) for t in F['terms']]:
        e = re.sub(r'S(\d+)', r'S[:, \1]', t).replace('max(0,', 'np.maximum(0,')
        ex.append((compile(e, t, 'eval'), c))
    units = sorted({int(u) for t in F['terms'] for u in re.findall(r'S(\d+)', t[0])})
    def f(S):
        return F['intercept'] + sum(c * eval(e, {'np': np, 'S': S}) for e, c in ex)
    def cols(S):
        return np.stack([eval(e, {'np': np, 'S': S}) * np.ones(len(S)) for e, c in ex], 1)
    return f, cols, units


def deciles(v_fit, v):
    """decile 0..9 by fit-sample quantiles; ties (e.g. many exact zeros) broken at random so deciles stay equal-sized"""
    rng = np.random.default_rng(0)
    eps = 1e-9 * (np.abs(v_fit).max() + 1)
    qf = np.quantile(v_fit + eps * rng.random(len(v_fit)), np.linspace(0, 1, 11)[1:-1])
    return np.searchsorted(qf, v + eps * rng.random(len(v)))


def boot_ratio(m, p, idx_sets):
    """ratio Σm/Σp for each bootstrap index set"""
    return np.array([m[i].sum() / p[i].sum() if abs(p[i].sum()) > 1e-12 else np.nan for i in idx_sets])


def ci(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) > 10 else [None, None]


def bh(p):
    p = np.asarray([1.0 if v is None else v for v in p], float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1).tolist()


def slope_by_decile(dec, m, p, rng, ctrl=None, top=(7, 8, 9)):
    """edit slope per decile with bootstrap CI; H0 slope(d) = slope(top): bootstrap + permutation p"""
    out = []
    topm = np.isin(dec, top)
    ti = np.flatnonzero(topm)
    tb = [rng.choice(ti, len(ti)) for _ in range(NB)]
    s_top = m[ti].sum() / p[ti].sum(); s_top_b = boot_ratio(m, p, tb)
    for d in range(10):
        di = np.flatnonzero(dec == d)
        rec = dict(decile=d + 1, n=int(len(di)))
        if len(di) < 30 or abs(p[di].sum()) < 1e-9:
            out.append(rec); continue
        db = [rng.choice(di, len(di)) for _ in range(NB)]
        sb = boot_ratio(m, p, db); s = m[di].sum() / p[di].sum()
        diff_b = sb - s_top_b
        pb = float(min(1, 2 * min((diff_b <= 0).mean(), (diff_b >= 0).mean()) + 1 / NB))
        # permutation: shuffle membership between decile d and the top group
        if d in top:
            pp = None
        else:
            u = np.r_[di, ti]; nd = len(di); obs = abs(s - s_top); cnt = 0
            for _ in range(NPERM):
                pr = rng.permutation(u); a, b = pr[:nd], pr[nd:]
                cnt += abs(m[a].sum() / p[a].sum() - m[b].sum() / p[b].sum()) >= obs
            pp = float((cnt + 1) / (NPERM + 1))
        ols = float(np.polyfit(p[di], m[di], 1)[0]) if p[di].std() > 1e-12 else None
        rec.update(slope=float(s), ci=ci(sb), ols_slope=ols, mean_pred=float(p[di].mean()), mean_meas=float(m[di].mean()),
                   diff_vs_top=float(s - s_top), diff_ci=ci(diff_b), p_boot=pb if d not in top else None, p_perm=pp)
        if ctrl is not None:
            cm = ctrl[di]
            cbs = np.array([ctrl[i].mean() for i in db]) / abs(p[di].mean())
            rec.update(control_mean=float(cm.mean()), control_over_pred=float(cm.mean() / abs(p[di].mean())), control_ci=ci(cbs))
        out.append(rec)
    return dict(top_slope=float(s_top), top_ci=ci(s_top_b), deciles=out)


def main():
    rng = np.random.default_rng(30)
    xf, yf, _ = sample(400 if QUICK else 2000, 3001); _ = None
    xt, yt, Qt = sample(320 if QUICK else 1600, 3002)
    hf = {hash(r.tobytes()) for r in xf.reshape(len(xf), -1)}
    new = np.array([hash(r.tobytes()) not in hf for r in xt.reshape(len(xt), -1)])
    xt, yt = xt[new], yt[new]; Qt = {k: v[new] for k, v in Qt.items()}
    log(f'fit jets {len(xf)} (seed 3001), test jets {len(xt)} (seed 3002, {int((~new).sum())} shared with fit removed)')
    net = Jedi(64); unt = Jedi(64, untrained=True, seed=0)
    _, KEEP = masks(net)
    R5 = {u['S']: u for u in json.loads((OUT / 'e4_e5.json').read_text())['E5']['units']}
    K = ('S', 'h5', 'z', 'logits')
    Bf = net.run(xf, keep=K); Bt = net.run(xt, keep=K)
    acc_f = float((Bf['logits'].argmax(1) == yf).mean()); acc_t = float((Bt['logits'].argmax(1) == yt).mean())
    log(f'accuracy unedited: fit {acc_f:.4f}, test {acc_t:.4f} (expected 0.806-0.813)')
    ct = Bt['logits'].argmax(1)
    W7, b7 = net.KB['d7']
    lo_float = np.maximum(Bt['z'], 0) @ W7 + b7
    log(f'check d7 in float on ReLU(z): max |Δlogit| {np.abs(lo_float - Bt["logits"]).max():.3g}, calls differ {(lo_float.argmax(1) != ct).mean()*100:.3f} %')
    chk = net.run_jetwise(xt[:500], {'S': (np.ones(64), Bt['S'][:500])}, keep=('z',))
    log(f'check S override with own S: max |Δz| {np.abs(chk["z"] - Bt["z"][:500]).max():.3g}')
    res = dict(N_fit=len(xf), N_test=len(xt), seed_fit=3001, seed_test=3002, acc_fit=acc_f, acc_test=acc_t,
               d7_float_maxdiff=float(np.abs(lo_float - Bt['logits']).max()), part1={}, part2={}, part3={})
    figdata = {}

    # ------------------------------------------------------------ Part 1 + 2 for S units
    Hf_all = handbuilt(xf); Ht_all = handbuilt(xt)
    for c in SU:
        u = R5[c]; keep = KEEP[:, c]
        Hf = dict(Hf_all); Hf.update(handbuilt(xf, keep, 'K:')); Ht = dict(Ht_all); Ht.update(handbuilt(xt, keep, 'K:'))
        Hf['hinge'] = hinge_sum(u['hinge'], u['f_pad'], keep, xf); Ht['hinge'] = hinge_sum(u['hinge'], u['f_pad'], keep, xt)
        names = sorted(Hf); Xf = np.stack([Hf[k] for k in names], 1); Xt = np.stack([Ht[k] for k in names], 1)
        named = {29: ['ΣpT·r^2', 'Σr^2', 'N(pT<2,r>0.1)'], 1: ['K:Σ(r-0.2)+'], 0: ['K:ΣpT']}[c]
        forms = dict(named=[names.index(k) for k in named], hinge=[names.index('hinge')])
        sf, st = Bf['S'][:, c], Bt['S'][:, c]; sd = float(sf.std())
        df, dt = deciles(sf, sf), deciles(sf, st); top_f = df >= 7
        P1 = dict(sd=sd, frac_zero_test=float((st <= 1e-9).mean()), decile_edges=np.quantile(sf, np.linspace(0, 1, 11)).tolist(), forms={})
        for fn, cols in forms.items():
            cf = linfit(Xf[top_f][:, cols], sf[top_f]); pt_ = np.c_[Xt[:, cols], np.ones(len(xt))] @ cf
            rows = []
            for d in range(10):
                mf, mt = df == d, dt == d
                cd = linfit(Xf[mf][:, cols], sf[mf]); pd_ = np.c_[Xt[mt][:, cols], np.ones(mt.sum())] @ cd
                Ff = greedy(Xf[mf], sf[mf], 3, names); pfree = apply(Ff, Xt[mt])
                tt = st[mt]
                g = rmse_gap(pt_[mt], pfree, tt, sd, rng)
                rows.append(dict(decile=d + 1, n=int(mt.sum()), R2_top=r2(pt_[mt], tt), R2_sameform=r2(pd_, tt), R2_free=r2(pfree, tt),
                                 rmse_top_sd=float(np.sqrt(((pt_[mt] - tt) ** 2).mean()) / sd),
                                 rmse_free_sd=float(np.sqrt(((pfree - tt) ** 2).mean()) / sd),
                                 bias_top_sd=float((pt_[mt] - tt).mean() / sd), free_terms=Ff['terms'],
                                 rmse_sameform_sd=float(np.sqrt(((pd_ - tt) ** 2).mean()) / sd), sameform_coef=cd[:-1].tolist(), gap_rmse_top_minus_free=float(g.mean()) if len(g) else None,
                                 gap_ci=ci(g), p_gap=float(min(1, 2 * min((g <= 0).mean(), (g >= 0).mean()) + 1 / 300)) if len(g) else None))
            P1['forms'][fn] = dict(terms=[names[k] for k in cols], top_coef=cf.tolist(), R2_top_all_test=r2(pt_, st), deciles=rows)
            log(f'S{c} [{fn}] top-fit R² all test {r2(pt_, st):.3f}; per decile R²top/free, rmse_top/free (sd): ' +
                ' '.join(f"{r['decile']}:{(r['R2_top'] or 0):.2f}/{(r['R2_free'] or 0):.2f},{r['rmse_top_sd']:.3f}/{r['rmse_free_sd']:.3f}" for r in rows))
            log(f'   free terms per decile: ' + ' | '.join(','.join(r['free_terms']) for r in rows))
            if fn == 'named': cf_named = cf
        res['part1'][f'S{c}'] = P1

        # ---------------- Part 2: edits on test jets
        J = len(xt); pt = xt[..., 0]; real = pt > 0; r = np.hypot(xt[..., 1], xt[..., 2]); slot = np.arange(64)[None, :]
        Jr = np.arange(J)
        if c in (29, 1):
            p = u['hinge']
            dr = rng.choice([0.01, 0.02, 0.03], J) * rng.choice([-1, 1], J)
            hb = hinge_eval(p, np.maximum(pt, 1e-3), r); ha = hinge_eval(p, np.maximum(pt, 1e-3), np.maximum(r + dr[:, None], 0))
            win = keep[None, :] & real & ((r + dr[:, None]) > 0.005)
            cand = win & ((hb > 0) | (ha > 0))
            cc = (real & ~keep[None, :]) if (~keep).any() else (win & (hb <= 0) & (ha <= 0))
            cc &= (r + dr[:, None]) > 0.005
            ok = cand.any(1) & cc.any(1)
            j = np.array([rng.choice(np.flatnonzero(o)) if o.any() else 0 for o in cand])
            jc = np.array([rng.choice(np.flatnonzero(o)) if o.any() else 0 for o in cc])
            xe = move_radial(xt, j, np.maximum(r[Jr, j] + dr, 0)); xc = move_radial(xt, jc, np.maximum(r[Jr, jc] + dr, 0))
            pred = hinge_sum(p, u['f_pad'], keep, xe) - hinge_sum(p, u['f_pad'], keep, xt)
            pred_c = hinge_sum(p, u['f_pad'], keep, xc) - hinge_sum(p, u['f_pad'], keep, xt)
            einfo = dict(edit='radial move of one particle at fixed pT and azimuth, Δr ∈ ±{0.01,0.02,0.03}',
                         control='slot 0 (outside the window)' if (~keep).any() else 'particle with hinge off before and after')
        else:
            f = rng.choice([0.98, 1.02], J)
            n = real.sum(1)
            j = rng.integers(0, 14, J); jc = np.array([rng.integers(14, max(nn, 15)) for nn in n])
            ok = n > 16
            xe = xt.copy(); xe[Jr, j, 0] *= f; xe = resort(xe)
            xc = xt.copy(); xc[Jr, jc, 0] *= f; xc = resort(xc)
            HKe = handbuilt(xe, keep, 'K:')['K:ΣpT']; HKc = handbuilt(xc, keep, 'K:')['K:ΣpT']
            pred = cf_named[0] * (HKe - Ht['K:ΣpT']); pred_c = cf_named[0] * (HKc - Ht['K:ΣpT'])
            einfo = dict(edit='pT × (1 ± 0.02) of one of the 14 hardest particles, ranks re-sorted; predicted = top-fit coef × Δ(ΣpT of slots 0-13)',
                         control='same nudge on a particle of rank ≥ 14', top_coef=float(cf_named[0]),
                         hinge_direct_pred_mean=float(np.abs(hinge_sum(u['hinge'], u['f_pad'], keep, xe) - hinge_sum(u['hinge'], u['f_pad'], keep, xt))[ok].mean()))
        ok &= in_range(net, xe) & in_range(net, xc)
        E = net.run(xe[ok], keep=('S', 'logits')); C = net.run(xc[ok], keep=('S',))
        Ug0 = unt.run(xt[ok], keep=('S',)); Ug1 = unt.run(xe[ok], keep=('S',))
        meas = E['S'][:, c] - st[ok]; meas_c = C['S'][:, c] - st[ok]; meas_u = Ug1['S'][:, c] - Ug0['S'][:, c]
        pr = pred[ok]; dec = dt[ok]
        log(f'S{c} edit: n {ok.sum()}, mean pred {pr.mean():+.4f} (|pred| {np.abs(pr).mean():.4f}), meas {meas.mean():+.4f}, '
            f'overall signed slope {(meas*np.sign(pr)).sum()/np.abs(pr).sum():.3f}; control pred {pred_c[ok].mean():+.4f} meas {meas_c.mean():+.4f}; untrained meas {meas_u.mean():+.4f}; calls changed {(E["logits"].argmax(1) != ct[ok]).mean()*100:.2f} %')
        # signed slopes: use pred*sign so that +/- edits add
        sg = np.sign(pr); sg[sg == 0] = 1
        SD = slope_by_decile(dec, meas * sg, pr * sg, rng, ctrl=meas_c * sg)
        ub = [rng.choice(len(pr), len(pr)) for _ in range(NB)]
        SD.update(einfo, n=int(ok.sum()), overall_slope=float((meas * sg).sum() / (pr * sg).sum()),
                  overall_ci=ci(boot_ratio(meas * sg, pr * sg, ub)),
                  control_pred_mean=float(pred_c[ok].mean()), control_meas_over_pred=float((meas_c * sg).mean() / np.abs(pr).mean()),
                  control_ci=ci([(meas_c * sg)[i].mean() / np.abs(pr).mean() for i in ub]),
                  untrained_slope=float((meas_u * sg).sum() / (pr * sg).sum()), untrained_ci=ci(boot_ratio(meas_u * sg, pr * sg, ub)),
                  calls_changed=float((E['logits'].argmax(1) != ct[ok]).mean()))
        res['part2'][f'S{c}'] = SD
        log(f'S{c} slope per decile: ' + ' '.join(f"{d['decile']}:{d.get('slope', float('nan')):.2f}" for d in SD['deciles']) +
            f" | top {SD['top_slope']:.3f} {SD['top_ci']}; control/pred {SD['control_meas_over_pred']:+.3f}; untrained {SD['untrained_slope']:+.3f}")
        if c == 29:   # example jet for the figure: a weak-range jet (decile 2) with an outward move
            w = np.flatnonzero(ok)[(dec == 1) & (pr > 0)]
            k = int(w[0]) if len(w) else int(np.flatnonzero(ok)[0])
            figdata.update(jet_x=xt[k], jet_j=int(j[k]), jet_dr=float(dr[k]), jet_label=int(yt[k]), jet_S=float(st[k]),
                           jet_pred=float(pred[k]), jet_meas=float(net.run(xe[k:k + 1], keep=('S',))['S'][0, c] - st[k]),
                           jet_decile=int(dt[k] + 1))

    # ------------------------------------------------------------ Part 1 + 2 for z units (positive side)
    Sf, St = Bf['S'][:, LIVE], Bt['S'][:, LIVE]
    knots = [np.unique(np.quantile(Sf[:, i], np.linspace(0.1, 0.9, 9))) for i in range(len(LIVE))]

    def basis(S):
        cols, nm = [], []
        for i, c in enumerate(LIVE):
            cols.append(S[:, i]); nm.append(f'S{c}')
            for t in knots[i]:
                cols.append(np.maximum(S[:, i] - t, 0)); nm.append(f'(S{c}-{t:.3g})+')
                cols.append(np.maximum(t - S[:, i], 0)); nm.append(f'({t:.3g}-S{c})+')
        return np.stack(cols, 1), nm
    Xf, nmz = basis(Sf); Xt, _ = basis(St)
    col_unit = np.array([LIVE.index(int(s.split('S')[1].split('-')[0].split(')')[0])) for s in nmz])
    E7 = json.loads((OUT / 'e7_S.json').read_text())['formulas']['32']

    for k in ZU:
        f7, cols7, units7 = e7_formula(E7[k])
        zf, zt = Bf['z'][:, k], Bt['z'][:, k]; pf, ptm = zf > 0, zt > 0
        sd = float(zf[pf].std())
        df = np.full(len(zf), -1); dtz = np.full(len(zt), -1)
        df[pf] = deciles(zf[pf], zf[pf]); dtz[ptm] = deciles(zf[pf], zt[ptm])
        top_f = df >= 7
        p7 = f7(Bt['S'])                                                       # E7 formula, not refit
        C7f, C7t = cols7(Bf['S']), cols7(Bt['S'])
        c7top = linfit(C7f[top_f], zf[top_f]); p7top = np.c_[C7t, np.ones(len(zt))] @ c7top   # E7 terms refit on top 3 deciles
        rows = []
        for d in range(10):
            mf, mt = df == d, dtz == d
            Ff = greedy(Xf[mf], zf[mf], 8, nmz); pfree = apply(Ff, Xt[mt]); tt = zt[mt]
            g = rmse_gap(p7[mt], pfree, tt, sd, rng); g2 = rmse_gap(p7top[mt], pfree, tt, sd, rng)
            rows.append(dict(decile=d + 1, n=int(mt.sum()), R2_e7=r2(p7[mt], tt), R2_e7top=r2(p7top[mt], tt), R2_free=r2(pfree, tt),
                             rmse_e7_sd=float(np.sqrt(((p7[mt] - tt) ** 2).mean()) / sd),
                             rmse_e7top_sd=float(np.sqrt(((p7top[mt] - tt) ** 2).mean()) / sd),
                             rmse_free_sd=float(np.sqrt(((pfree - tt) ** 2).mean()) / sd),
                             bias_e7_sd=float((p7[mt] - tt).mean() / sd), bias_e7top_sd=float((p7top[mt] - tt).mean() / sd),
                             free_terms=Ff['terms'], free_units=sorted({f'S{LIVE[col_unit[c_]]}' for c_ in Ff['cols']}),
                             gap_rmse_top_minus_free=float(g.mean()), gap_ci=ci(g), p_gap=float(min(1, 2 * min((g <= 0).mean(), (g >= 0).mean()) + 1 / 300)),
                             gap_e7top_minus_free=float(g2.mean()), gap_e7top_ci=ci(g2)))
        res['part1'][f'z{k}'] = dict(sd_pos=sd, frac_pos_test=float(ptm.mean()), e7_units=[f'S{u}' for u in units7], e7_nterms=len(E7[k]['terms']),
                                     R2_e7_pos_test=r2(p7[ptm], zt[ptm]), R2_e7top_pos_test=r2(p7top[ptm], zt[ptm]), deciles=rows)
        log(f'z{k} E7 formula (K=32, units {units7}) R² positive test {r2(p7[ptm], zt[ptm]):.3f}, refit on top-3 deciles {r2(p7top[ptm], zt[ptm]):.3f}; '
            'per decile rmse e7/e7top/free (sd), bias e7: ' +
            ' '.join(f"{r['decile']}:{r['rmse_e7_sd']:.2f}/{r['rmse_e7top_sd']:.2f}/{r['rmse_free_sd']:.2f},{r['bias_e7_sd']:+.2f}" for r in rows))
        log('   free units per decile: ' + ' | '.join(','.join(r['free_units']) for r in rows))

        # Part 2 for z: S-level intervention (central difference ± 0.5 sd) on the S unit with the largest E7-predicted effect
        sel = ptm; S0_ = Bt['S'][sel]
        def dpred(c64, delta):
            Sp = S0_.copy(); Sp[:, c64] += delta; Sm = S0_.copy(); Sm[:, c64] -= delta
            return f7(Sp) - f7(Sm)
        sdS = Bf['S'].std(0)
        eff = {c64: float(np.abs(dpred(c64, 0.5 * sdS[c64])).mean()) for c64 in LIVE}
        ca = max(eff, key=eff.get); cc_ = min((c64 for c64 in LIVE if c64 not in units7), key=lambda c64: eff[c64], default=min(eff, key=eff.get))
        dS = 0.5 * float(sdS[ca]); dSc = 0.5 * float(sdS[cc_])
        def runS(nt, base, c64, delta):
            V = base.copy(); V[:, c64] += delta; m = np.zeros(64); m[c64] = 1
            return nt.run_jetwise(xt[sel], {'S': (m, V)}, keep=('z',))['z'][:, k]
        meas = runS(net, S0_, ca, dS) - runS(net, S0_, ca, -dS)
        pred = dpred(ca, dS)
        meas_c = runS(net, S0_, cc_, dSc) - runS(net, S0_, cc_, -dSc)
        Su = unt.run(xt[sel], keep=('S',))['S']
        meas_u = runS(unt, Su, ca, dS) - runS(unt, Su, ca, -dS)
        dec = dtz[sel]
        SD = slope_by_decile(dec, meas, pred, rng, ctrl=meas_c)
        ub = [rng.choice(len(pred), len(pred)) for _ in range(NB)]
        SD.update(edit=f'S{ca} set to S{ca} ± {dS:.3f} (0.5 sd), central difference; predicted Δz from the E7 K=32 formula',
                  control=f'S{cc_} ± {dSc:.3f} (absent from the formula; E7-predicted |Δz| {eff[cc_]:.3g})',
                  edited_unit=f'S{ca}', control_unit=f'S{cc_}', predicted_abs_effect_by_unit={f'S{c}': v for c, v in eff.items()},
                  n=int(sel.sum()), overall_slope=float(meas.sum() / pred.sum()),
                  overall_ci=ci(boot_ratio(meas, pred, ub)), control_meas_over_pred=float(meas_c.mean() / np.abs(pred).mean()),
                  control_ci=ci([meas_c[i].mean() / np.abs(pred).mean() for i in ub]),
                  untrained_slope=float(meas_u.sum() / pred.sum()), untrained_ci=ci(boot_ratio(meas_u, pred, ub)),
                  corr_meas_pred=float(np.corrcoef(meas, pred)[0, 1]))
        res['part2'][f'z{k}'] = SD
        log(f'z{k} S-edit on S{ca} (corr {SD["corr_meas_pred"]:.2f}): overall slope {SD["overall_slope"]:.3f} {SD["overall_ci"]}; per decile ' +
            ' '.join(f"{d['decile']}:{d.get('slope', float('nan')):.2f}" for d in SD['deciles']) +
            f" | control S{cc_} {SD['control_meas_over_pred']:+.3f}; untrained {SD['untrained_slope']:+.3f}")

    # ------------------------------------------------------------ Part 3: negative poles
    mass = np.asarray(Qt['mass'], float)
    yt_oh = np.eye(5)[yt]
    from scipy.stats import spearmanr
    def mcnemar(a, b):
        from scipy.stats import binomtest
        n01 = int((a & ~b).sum()); n10 = int((~a & b).sum())
        return dict(n_lost=n01, n_gained=n10, p=float(binomtest(n01, n01 + n10).pvalue) if n01 + n10 else 1.0)
    corr0 = ct == yt
    H5 = Bt['h5']
    for k in ZU:
        zt = Bt['z'][:, k]; zf = Bf['z'][:, k]; neg = zt < 0; negf = zf < 0
        o = np.argsort(zt); n1 = max(1, len(zt) // 100); n10 = len(zt) // 10
        mix = lambda ii: (np.bincount(yt[ii], minlength=5) / len(ii) * 100).round(1).tolist()
        rho = {f'S{c}': float(spearmanr(St[neg, i], zt[neg]).correlation) for i, c in enumerate(LIVE) if St[neg, i].std() > 0}
        Fn = greedy(Xf[negf], zf[negf], 8, nmz); pn = apply(Fn, Xt[neg])
        from sklearn.metrics import roc_auc_score
        auc = {cl: float(roc_auc_score(yt[neg] == i, -zt[neg])) if 0 < (yt[neg] == i).sum() < neg.sum() else None
               for i, cl in enumerate(['g', 'q', 'W', 'Z', 't'])}
        P3 = dict(frac_neg=float(neg.mean()), most_neg_1pct_mix=mix(o[:n1]), most_neg_10pct_mix=mix(o[:n10]), all_mix=mix(np.arange(len(yt))),
                  mass_median_most_neg_10pct=float(np.median(mass[o[:n10]])), mass_median_pos=float(np.median(mass[zt > 0])),
                  mass_median_neg=float(np.median(mass[neg])), z_q01=float(np.quantile(zt, 0.01)), z_sd_pos=float(zt[zt > 0].std()),
                  spearman_S_neg=dict(sorted(rho.items(), key=lambda kv: -abs(kv[1]))[:6]),
                  neg_fit_terms=Fn['terms'], neg_fit_coef=Fn['coef'], neg_fit_R2_test=r2(pn, zt[neg]), auc_minus_z_within_neg=auc)
        # mass deciles of negative side: mean z per mass bin
        mb = np.quantile(mass, np.linspace(0, 1, 11))
        P3['z_mean_by_mass_decile'] = [float(zt[(mass >= a) & (mass <= b)].mean()) for a, b in zip(mb[:-1], mb[1:])]
        P3['mass_decile_edges'] = mb.tolist()
        P3['bypass'] = {}
        for alpha in (1.0, 0.5):
            lo = Bt['logits'] + alpha * np.minimum(zt, 0)[:, None] * W7[k][None, :]
            cb = lo.argmax(1); ch = cb != ct
            ub = [rng.choice(len(yt), len(yt)) for _ in range(NB)]
            flows = {}
            for a_, b_ in zip(ct[ch], cb[ch]): flows[f'{a_}->{b_}'] = flows.get(f'{a_}->{b_}', 0) + 1
            P3['bypass'][str(alpha)] = dict(calls_changed=float(ch.mean()), calls_changed_ci=ci([ch[i].mean() for i in ub]),
                                            acc=float((cb == yt).mean()), dacc=float((cb == yt).mean() - corr0.mean()),
                                            dacc_ci=ci([(cb == yt)[i].mean() - corr0[i].mean() for i in ub]),
                                            mcnemar=mcnemar(corr0, cb == yt), flows_netcall_to_bypass=dict(sorted(flows.items(), key=lambda kv: -kv[1])[:8]))
        # control: 20 random h5 directions matched to z_k's negative fraction and sd
        q = neg.mean(); sdz = zt.std(); ctrl = []
        for t in range(20):
            v = rng.normal(size=H5.shape[1]); v /= np.linalg.norm(v)
            a = H5 @ v; a = (a - np.quantile(a, q)); a = a / a.std() * sdz
            lo = Bt['logits'] + np.minimum(a, 0)[:, None] * W7[k][None, :]; cb = lo.argmax(1)
            ctrl.append(dict(calls_changed=float((cb != ct).mean()), dacc=float((cb == yt).mean() - corr0.mean())))
        cc = np.array([c_['calls_changed'] for c_ in ctrl]); da = np.array([c_['dacc'] for c_ in ctrl])
        b1 = P3['bypass']['1.0']
        P3['random_h5_control'] = dict(calls_changed_mean=float(cc.mean()), calls_changed_range=[float(cc.min()), float(cc.max())],
                                       dacc_mean=float(da.mean()), dacc_range=[float(da.min()), float(da.max())],
                                       p_calls=float(((cc >= b1['calls_changed']).sum() + 1) / 21), p_dacc_lower=float(((da <= b1['dacc']).sum() + 1) / 21),
                                       p_dacc_higher=float(((da >= b1['dacc']).sum() + 1) / 21))
        res['part3'][f'z{k}'] = P3
        log(f'z{k} negative: {neg.mean()*100:.1f} % of jets; most negative 1 % (g q W Z t) {P3["most_neg_1pct_mix"]}; 10 % {P3["most_neg_10pct_mix"]}; '
            f'mass median most-neg-10 % {P3["mass_median_most_neg_10pct"]:.1f} vs z>0 {P3["mass_median_pos"]:.1f} GeV; neg fit R² {P3["neg_fit_R2_test"]:.3f} terms {Fn["terms"][:4]}')
        log(f'   AUC(-z) within z<0 {auc}; Spearman {P3["spearman_S_neg"]}')
        log(f'   bypass α=1: calls changed {b1["calls_changed"]*100:.2f} % {b1["calls_changed_ci"]}, Δacc {b1["dacc"]*100:+.2f} {b1["dacc_ci"]}, McNemar {b1["mcnemar"]}, flows {b1["flows_netcall_to_bypass"]}; '
            f'random h5: calls {cc.mean()*100:.2f} % [{cc.min()*100:.2f}, {cc.max()*100:.2f}], Δacc {da.mean()*100:+.2f} [{da.min()*100:+.2f}, {da.max()*100:+.2f}]')

    # ------------------------------------------------------------ BH
    keys2 = [(u, d) for u in res['part2'] for d in res['part2'][u]['deciles'] if d.get('p_perm') is not None]
    q2 = bh([d['p_perm'] for _, d in keys2])
    for (u, d), qq in zip(keys2, q2): d['q_BH_perm'] = qq
    q2b = bh([d['p_boot'] for _, d in keys2])
    for (u, d), qq in zip(keys2, q2b): d['q_BH_boot'] = qq
    res['n_tests_part2'] = len(keys2)
    keys1 = []
    for u, P in res['part1'].items():
        rows = P['forms']['named']['deciles'] if 'forms' in P else P['deciles']
        keys1 += [(u, r) for r in rows if r.get('p_gap') is not None]
    q1 = bh([r['p_gap'] for _, r in keys1])
    for (u, r), qq in zip(keys1, q1): r['q_BH'] = qq
    res['n_tests_part1'] = len(keys1)
    log(f'BH: part 2 {len(keys2)} tests, q<0.05: ' + ', '.join(f"{u} d{d['decile']} ({d['slope']:.2f} vs top {res['part2'][u]['top_slope']:.2f})" for (u, d), qq in zip(keys2, q2) if qq < 0.05))
    log(f'BH: part 1 {len(keys1)} gap tests, q<0.05: ' + ', '.join(f"{u} d{r['decile']} rmse gap {r['gap_rmse_top_minus_free']:+.3f}" for (u, r), qq in zip(keys1, q1) if qq < 0.05))

    suffix = '_quick' if QUICK else ''
    (OUT / f'e30_weak_poles{suffix}.json').write_text(json.dumps(res, indent=1, default=float))
    np.savez(OUT / f'e30_weak_poles{suffix}.npz', **{k: np.asarray(v) for k, v in figdata.items()})
    try:
        figure(res, figdata, OUT / f'e30_weak_poles{suffix}.png', R5[29]['hinge'])
    except Exception as e:
        log('figure failed', repr(e))
    log('done')


def figure(res, fd, path, hp):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw=dict(width_ratios=[1.5, 1]))
    lab = {'S29': 'S29  Σ pT-weighted r² hinge', 'S1': 'S1  Σ(r − 0.2)₊, ranks ≥ 1', 'S0': 'S0  ΣpT of 14 hardest',
           'z0': 'z0  ~75 GeV window', 'z7': 'z7  94–98 GeV window', 'z10': 'z10  135–148 GeV window'}
    cols = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    for (u, P), c in zip(res['part2'].items(), cols):
        D = [d for d in P['deciles'] if 'slope' in d]
        x = np.array([d['decile'] for d in D]) + (cols.index(c) - 2.5) * 0.06
        y = np.array([d['slope'] for d in D]); lo = np.array([d['ci'][0] for d in D]); hi = np.array([d['ci'][1] for d in D])
        ax[0].errorbar(x, y, yerr=[y - lo, hi - y], color=c, marker='o', ms=4, lw=1.4, capsize=2, label=lab.get(u, u))
    ax[0].axhspan(0.9, 1.1, color='0.85', zorder=0, label='0.9–1.1 (one job)')
    ax[0].axhline(0, color='0.5', lw=0.8)
    ax[0].set_xlabel('activation decile of the unedited unit (1 = weakest, 10 = strongest)')
    ax[0].set_ylabel('edit slope = measured Δ / predicted Δ')
    ax[0].set_xticks(range(1, 11)); ax[0].legend(fontsize=7.5, loc='lower right', ncol=2)
    ax[0].set_title('Edit slope across the activation range', loc='left', fontsize=11)
    if 'jet_x' in fd:
        x = fd['jet_x']; real = x[:, 0] > 0; j = int(fd['jet_j'])
        a = ax[1]
        a.scatter(x[real, 1], x[real, 2], s=8 + 3 * np.sqrt(x[real, 0]) * 8, c='0.6', alpha=0.6, edgecolor='none')
        r0 = np.hypot(x[j, 1], x[j, 2]); th = np.arctan2(x[j, 2], x[j, 1]); r1 = r0 + float(fd['jet_dr'])
        a.annotate('', xy=(r1 * np.cos(th), r1 * np.sin(th)), xytext=(x[j, 1], x[j, 2]), arrowprops=dict(arrowstyle='->', color='C3', lw=1.8))
        a.scatter([x[j, 1]], [x[j, 2]], s=60, facecolor='none', edgecolor='C3', lw=1.5)
        rb = -(hp['c'] + hp['a'] * np.log(max(x[j, 0], 1e-3))) / hp['b']
        t = np.linspace(0, 2 * np.pi, 200); a.plot(rb * np.cos(t), rb * np.sin(t), 'C3--', lw=0.8)
        a.set_aspect('equal'); a.set_xlabel('Δη to jet axis'); a.set_ylabel('Δφ to jet axis')
        a.set_title(f"S29 edit on one test jet (decile {fd['jet_decile']}, class {'gqWZt'[int(fd['jet_label'])]})\n"
                    f"particle pT {x[j,0]:.1f} GeV moved r {r0:.3f} → {r1:.3f}; dashed: hinge onset r for that pT\n"
                    f"predicted ΔS29 {fd['jet_pred']:+.3f}, measured {fd['jet_meas']:+.3f}", loc='left', fontsize=9)
        lim = max(0.45, r1 + 0.05); a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_jet():
    """--fig: one test jet (seed 3002) in S29 decile 2-3 with a particle on the hinge moved outward by 0.03 rad at fixed pT;
    saves the jet, the edit, the hinge prediction and the network's ΔS29 (fixed point: S steps of 1/16 per unit of u)"""
    xt, yt, _ = sample(1600, 3002)
    net = Jedi(64); _, KEEP = masks(net)
    u = {v['S']: v for v in json.loads((OUT / 'e4_e5.json').read_text())['E5']['units']}[29]; p = u['hinge']
    edges = json.loads((OUT / 'e30_weak_poles.json').read_text())['part1']['S29']['decile_edges']
    S = net.run(xt, keep=('S',))['S'][:, 29]
    pt = xt[..., 0]; r = np.hypot(xt[..., 1], xt[..., 2])
    on = (pt > 0) & (hinge_eval(p, np.maximum(pt, 1e-3), r) > 0) & (r > 0.08)
    for k in np.flatnonzero((S > edges[1]) & (S < edges[3]) & on.any(1)):
        for j in np.flatnonzero(on[k]):
            xe = move_radial(xt[k:k + 1], np.array([j]), np.array([r[k, j] + 0.03]))
            pr = float(hinge_sum(p, u['f_pad'], KEEP[:, 29], xe)[0] - hinge_sum(p, u['f_pad'], KEEP[:, 29], xt[k:k + 1])[0])
            me = float(net.run(xe, keep=('S',))['S'][0, 29] - S[k])
            if me > 0:
                np.savez(OUT / 'e30_fig_jet.npz', jet_x=xt[k], jet_j=j, jet_dr=0.03, jet_label=int(yt[k]), jet_S=float(S[k]),
                         jet_pred=pr, jet_meas=me, jet_decile=int(np.searchsorted(edges[1:-1], S[k]) + 1), hinge=np.array([p['a'], p['b'], p['c']]))
                log(f'jet {k} class {yt[k]} S29 {S[k]:.3f} particle {j} pT {pt[k, j]:.2f} r {r[k, j]:.3f} -> +0.03: pred {pr:+.4f} meas {me:+.4f}')
                return


if __name__ == '__main__':
    fig_jet() if '--fig' in sys.argv else main()
