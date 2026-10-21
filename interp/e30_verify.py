"""E30 verification (fresh sample, seed 4030): rerun of the key parts of e30_weak_poles_jedi.py plus missing controls.

Network: JEDI-linear 3-feature n64, fixed point (interp.jedi_core.Jedi(64)). Jets: sample(PER, 4030) (default 1,500 per class).
Deciles: of the unedited unit over this sample (ties at zero broken at random, as E30).

A. One-particle edits, edit slope per decile (Σ measured ΔS / Σ predicted ΔS, signs aligned), as E30 Part 2:
   S29, S1: one in-window particle whose E5 hinge is on before or after is moved radially by Δr ∈ ±{0.01, 0.02, 0.03} rad at
            fixed pT and azimuth; predicted ΔS = Δ(E5 hinge sum).
            added: per-jet OLS slope (measured on predicted) per decile; the slope split into 'on both before and after' and
            'crossing the onset' edits; in-window control = the same Δr on an in-window particle with hinge off before and
            after (for S1 E30 used the rank-0 particle, which the zero-bit mask removes, so its 0 was structural).
   S0:      one of the 14 hardest pT × (1 ± 0.02), re-sorted; predicted = (a/16) × Δ(ΣpT of slots 0-13), a = E5 direct
            coefficient (0.32); control = same nudge on rank ≥ 14.
   tests:   each decile vs all other deciles (2,000-shuffle permutation), BH over 3 units × 10 deciles; global heterogeneity
            T = Σ_d n_d (s_d − s)² with 2,000 decile-label permutations.
   check:   the edit changes exactly one particle (radial: η, φ of one slot; S0: one pT, multiset otherwise equal).
B. Negative side of z0, z7, z10: ReLU bypass logits + min(z_k, 0)·W7[k] (α = 1); controls: 20 random h5 directions scaled to
   z_k's negative fraction and sd (E30's control), 20 global permutations of min(z_k, 0) across jets (same values, no
   per-jet information), 20 within-class permutations (keeps only class-level information).
C. Causal mass test of the negative side: angle edit Δη, Δφ × k (k = 1 ± 0.05; jet mass × k, M0 fixed), central difference
   dz_k / d ln m per mass bin and sign of z_k. Controls: iso-mass edit (angles × k, pT × 1/k: mass unchanged, same angular
   displacement), rotation about the jet axis by ±0.05 rad (mass and every r unchanged; displacement 0.05 r, as the angle edit).
   Calibration: slope of the binned mean of z_k against mean ln m over the unedited jets (the 'descriptive' curve E30 used).
Output: $AI_RESULTS/jedi/e30_verify.json.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.jedi_core import Jedi, sample
from interp.e4_e5_jedi import hinge_sum, hinge_eval, masks
from interp.e5_causal_jedi import in_range, resort, move_radial
from interp.e30_weak_poles_jedi import deciles, ci, bh, boot_ratio

OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'jedi'; OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
PER = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1500
SEED = 4030
NB = 1000; NPERM = 2000
SU = [29, 1, 0]; ZU = [0, 7, 10]
MBINS = [0, 40, 60, 80, 90, 100, 120, 150, 200, 1000]


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def mass(x):
    x = np.asarray(x, np.float64); pt, eta, phi = x[..., 0], x[..., 1], x[..., 2]
    p4 = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)], -1).sum(1)
    return np.sqrt(np.maximum(p4[:, 3] ** 2 - (p4[:, :3] ** 2).sum(1), 0.0))


def ols(m, p):
    v = ((p - p.mean()) ** 2).sum()
    return float(((p - p.mean()) * (m - m.mean())).sum() / v) if v > 1e-15 else np.nan


def slope_table(dec, m, p, rng, ctrls=None, sub=None):
    """per decile: ratio slope + CI, OLS slope + CI, test vs all other deciles (permutation); global heterogeneity test"""
    s_all = m.sum() / p.sum(); rows = []
    for d in range(10):
        di = np.flatnonzero(dec == d); oi = np.flatnonzero(dec != d)
        rec = dict(decile=d + 1, n=int(len(di)))
        if len(di) < 30 or abs(p[di].sum()) < 1e-9:
            rows.append(rec); continue
        db = [rng.choice(di, len(di)) for _ in range(NB)]
        s = m[di].sum() / p[di].sum(); s_o = m[oi].sum() / p[oi].sum()
        u = np.r_[di, oi]; nd = len(di); cnt = 0; obs = abs(s - s_o)
        for _ in range(NPERM):
            pr = rng.permutation(u); a, b = pr[:nd], pr[nd:]
            cnt += abs(m[a].sum() / p[a].sum() - m[b].sum() / p[b].sum()) >= obs
        rec.update(slope=float(s), ci=ci(boot_ratio(m, p, db)), slope_others=float(s_o), p_perm_vs_others=float((cnt + 1) / (NPERM + 1)),
                   ols=ols(m[di], p[di]), ols_ci=ci([ols(m[i], p[i]) for i in db[:300]]))
        for nm, c in (ctrls or {}).items():
            rec[f'ctrl_{nm}_over_pred'] = float(c[di].mean() / np.abs(p[di]).mean())
            rec[f'ctrl_{nm}_ci'] = ci([c[i].mean() / np.abs(p[i]).mean() for i in db[:300]])
        for nm, msk in (sub or {}).items():
            k = di[msk[di]]
            if len(k) >= 20 and abs(p[k].sum()) > 1e-9:
                kb = [rng.choice(k, len(k)) for _ in range(300)]
                rec[f'slope_{nm}'] = float(m[k].sum() / p[k].sum()); rec[f'slope_{nm}_ci'] = ci(boot_ratio(m, p, kb)); rec[f'n_{nm}'] = int(len(k))
        rows.append(rec)
    # global heterogeneity
    def T(lab):
        M = np.bincount(lab, m, 10); P = np.bincount(lab, p, 10); n = np.bincount(lab, minlength=10)
        ok = np.abs(P) > 1e-9
        return float((n[ok] * (M[ok] / P[ok] - s_all) ** 2).sum())
    t_obs = T(dec); tp = np.array([T(rng.permutation(dec)) for _ in range(NPERM)])
    ub = [rng.choice(len(m), len(m)) for _ in range(NB)]
    return dict(overall=float(s_all), overall_ci=ci(boot_ratio(m, p, ub)), overall_ols=ols(m, p),
                heterogeneity_T=t_obs, heterogeneity_p=float(((tp >= t_obs).sum() + 1) / (NPERM + 1)), deciles=rows)


def main():
    rng = np.random.default_rng(31)
    x, y, Q = sample(PER, SEED)
    log(f'jets {len(x)} (sample({PER}, {SEED}))')
    net = Jedi(64)
    _, KEEP = masks(net)
    R5 = {u['S']: u for u in json.loads((OUT / 'e4_e5.json').read_text())['E5']['units']}
    B = net.run(x, keep=('S', 'h5', 'z', 'logits'))
    ct = B['logits'].argmax(1); acc = float((ct == y).mean())
    log(f'accuracy {acc:.4f}')
    res = dict(N=len(x), per_class=PER, seed=SEED, acc=acc, A={}, B={}, C={})
    J = len(x); pt = x[..., 0]; real = pt > 0; r = np.hypot(x[..., 1], x[..., 2]); Jr = np.arange(J)

    # ------------------------------------------------------------------ A
    for c in SU:
        u = R5[c]; keep = KEEP[:, c]; st = B['S'][:, c]; dec = deciles(st, st)
        if c in (29, 1):
            p = u['hinge']
            dr = rng.choice([0.01, 0.02, 0.03], J) * rng.choice([-1, 1], J)
            hb = hinge_eval(p, np.maximum(pt, 1e-3), r); ha = hinge_eval(p, np.maximum(pt, 1e-3), np.maximum(r + dr[:, None], 0))
            win = keep[None, :] & real & ((r + dr[:, None]) > 0.005)
            cand = win & ((hb > 0) | (ha > 0))
            cin = win & (hb <= 0) & (ha <= 0)                            # in-window, hinge off before and after
            cout = real & ~keep[None, :] & ((r + dr[:, None]) > 0.005)   # outside the rank window (S1: slot 0)
            ok = cand.any(1) & cin.any(1)
            pick = lambda M: np.array([rng.choice(np.flatnonzero(o)) if o.any() else 0 for o in M])
            j = pick(cand); jin = pick(cin)
            xe = move_radial(x, j, np.maximum(r[Jr, j] + dr, 0)); xin = move_radial(x, jin, np.maximum(r[Jr, jin] + dr, 0))
            nchg = ((xe != x).any(2)).sum(1); nchg_in = ((xin != x).any(2)).sum(1)
            pred = hinge_sum(p, u['f_pad'], keep, xe) - hinge_sum(p, u['f_pad'], keep, x)
            pred_in = hinge_sum(p, u['f_pad'], keep, xin) - hinge_sum(p, u['f_pad'], keep, x)
            both = (hb[Jr, j] > 0) & (ha[Jr, j] > 0)
            if cout.any():
                ok &= cout.any(1); jout = pick(cout); xout = move_radial(x, jout, np.maximum(r[Jr, jout] + dr, 0))
            else:
                xout = None
            ok &= in_range(net, xe) & in_range(net, xin) & (in_range(net, xout) if xout is not None else True)
            extra = dict(edited_particles_per_jet=np.bincount(nchg[ok], minlength=3).tolist(), control_particles_per_jet=np.bincount(nchg_in[ok], minlength=3).tolist(),
                         pred_in_window_control_max_abs=float(np.abs(pred_in[ok]).max()), frac_crossing=float((~both[ok]).mean()),
                         dr_onset_mean_by_decile=None)
            # distance of the edited particle to the hinge onset (in hinge units / b = rad), by decile
            don = (p['a'] * np.log(np.maximum(pt[Jr, j], 1e-3)) + p['b'] * r[Jr, j] + p['c']) / p['b']   # r − r_onset (rad)
            extra['dr_onset_mean_by_decile'] = [float(np.median(don[ok][dec[ok] == d])) for d in range(10)]
        else:
            f = rng.choice([0.98, 1.02], J); n = real.sum(1)
            j = rng.integers(0, 14, J); jc = np.array([rng.integers(14, max(nn, 15)) for nn in n])
            ok = n > 16
            xe = x.copy(); xe[Jr, j, 0] *= f; xe = resort(xe)
            xin = x.copy(); xin[Jr, jc, 0] *= f; xin = resort(xin)
            Kp = lambda xx: (xx[..., 0] * (xx[..., 0] > 0) * keep[None, :]).sum(1)
            a16 = u['hinge']['a'] / 16
            pred = a16 * (Kp(xe) - Kp(x)); pred_in = a16 * (Kp(xin) - Kp(x))
            same = np.isclose(np.sort(xe[..., 0], 1), np.sort(x[..., 0], 1)).sum(1)
            ok &= in_range(net, xe) & in_range(net, xin)
            xout = None; both = np.ones(J, bool)
            extra = dict(pT_values_unchanged_per_jet_min=int(same[ok].min()), coef_per_GeV=float(a16),
                         control_pred_mean_abs=float(np.abs(pred_in[ok]).mean()))
        E = net.run(xe[ok], keep=('S',))['S'][:, c] - st[ok]
        Cin = net.run(xin[ok], keep=('S',))['S'][:, c] - st[ok]
        ctr = {'in_window_off' if c != 0 else 'rank_ge14': Cin}
        if xout is not None:
            ctr['outside_window'] = net.run(xout[ok], keep=('S',))['S'][:, c] - st[ok]
        pr = pred[ok]; sg = np.sign(pr); sg[sg == 0] = 1
        ctr = {k: v * sg for k, v in ctr.items()}
        sub = dict(on_both=both[ok], crossing=~both[ok]) if c in (29, 1) else None
        T = slope_table(dec[ok], E * sg, pr * sg, rng, ctrls=ctr, sub=sub)
        T.update(extra, n=int(ok.sum()), control_over_pred={k: float(v.mean() / np.abs(pr).mean()) for k, v in ctr.items()},
                 control_ci={k: ci([v[i].mean() / np.abs(pr[i]).mean() for i in [rng.choice(len(pr), len(pr)) for _ in range(300)]]) for k, v in ctr.items()})
        if c in (29, 1):
            for nm, msk in sub.items():
                T[f'overall_{nm}'] = float((E * sg)[msk].sum() / (pr * sg)[msk].sum())
        res['A'][f'S{c}'] = T
        log(f'S{c}: n {ok.sum()}, overall {T["overall"]:.3f} {T["overall_ci"]}, OLS {T["overall_ols"]:.3f}, heterogeneity p {T["heterogeneity_p"]:.4f}; controls {T["control_over_pred"]}')
        log('   deciles slope [ci] / OLS / p_vs_others: ' + ' | '.join(
            f"{d['decile']}:{d.get('slope', np.nan):.2f} [{(d.get('ci') or [np.nan])[0]:.2f},{(d.get('ci') or [np.nan, np.nan])[1]:.2f}] /{d.get('ols', np.nan):.2f} /{d.get('p_perm_vs_others', np.nan):.4f}" for d in T['deciles']))
        if sub:
            log('   on-both / crossing slope: ' + ' | '.join(f"{d['decile']}:{d.get('slope_on_both', np.nan):.2f}({d.get('n_on_both', 0)})/{d.get('slope_crossing', np.nan):.2f}({d.get('n_crossing', 0)})" for d in T['deciles']))
        log('   extra ' + json.dumps({k: v for k, v in extra.items()}))
    keys = [(u_, d) for u_ in res['A'] for d in res['A'][u_]['deciles'] if 'p_perm_vs_others' in d]
    for (u_, d), q in zip(keys, bh([d['p_perm_vs_others'] for _, d in keys])): d['q_BH'] = q
    log('BH q<0.05: ' + ', '.join(f"{u_} d{d['decile']} {d['slope']:.2f} vs others {d['slope_others']:.2f} (q {d['q_BH']:.4f})" for u_, d in keys if d['q_BH'] < 0.05))

    # ------------------------------------------------------------------ B
    W7, b7 = net.KB['d7']; corr0 = ct == y; H5 = B['h5']
    for k in ZU:
        z = B['z'][:, k]; neg = z < 0; zn = np.minimum(z, 0)
        def evalb(add):
            cb = (B['logits'] + add[:, None] * W7[k][None, :]).argmax(1)
            return float((cb != ct).mean()), float((cb == y).mean() - corr0.mean()), cb
        cc1, da1, cb1 = evalb(zn)
        ub = [rng.choice(J, J) for _ in range(NB)]
        n01 = int((corr0 & (cb1 != y)).sum()); n10 = int((~corr0 & (cb1 == y)).sum())
        from scipy.stats import binomtest
        P = dict(frac_neg=float(neg.mean()), bypass=dict(calls_changed=cc1, dacc=da1, dacc_ci=ci([((cb1 == y)[i].mean() - corr0[i].mean()) for i in ub]),
                                                        mcnemar=dict(n_lost=n01, n_gained=n10, p=float(binomtest(n01, n01 + n10).pvalue) if n01 + n10 else 1.0)),
                 median_mass_neg=float(np.median(Q['mass'][neg])), median_mass_pos=float(np.median(Q['mass'][~neg])),
                 mix_most_neg_10pct=(np.bincount(y[np.argsort(z)[:J // 10]], minlength=5) / (J // 10) * 100).round(1).tolist())
        ctrl = {}
        q = neg.mean(); sdz = z.std(); L = []
        for t in range(20):
            v = rng.normal(size=H5.shape[1]); v /= np.linalg.norm(v)
            a = H5 @ v; a = a - np.quantile(a, q); a = a / a.std() * sdz
            L.append(evalb(np.minimum(a, 0))[:2])
        ctrl['random_h5'] = L
        ctrl['perm_global'] = [evalb(zn[rng.permutation(J)])[:2] for _ in range(20)]
        L = []
        for _ in range(20):
            zp = zn.copy()
            for cl in range(5):
                ii = np.flatnonzero(y == cl); zp[ii] = zn[rng.permutation(ii)]
            L.append(evalb(zp)[:2])
        ctrl['perm_within_class'] = L
        for nm, L in ctrl.items():
            A = np.array(L)
            P[nm] = dict(calls_mean=float(A[:, 0].mean()), calls_range=[float(A[:, 0].min()), float(A[:, 0].max())],
                         dacc_mean=float(A[:, 1].mean()), dacc_range=[float(A[:, 1].min()), float(A[:, 1].max())],
                         p_dacc_lower=float(((A[:, 1] <= da1).sum() + 1) / 21), p_calls_higher=float(((A[:, 0] >= cc1).sum() + 1) / 21))
        res['B'][f'z{k}'] = P
        log(f'z{k}: neg {neg.mean()*100:.1f} %; bypass calls {cc1*100:.2f} %, dacc {da1*100:+.2f} {P["bypass"]["dacc_ci"]}, McNemar {P["bypass"]["mcnemar"]}; '
            + '; '.join(f"{nm} dacc {P[nm]['dacc_mean']*100:+.2f} [{P[nm]['dacc_range'][0]*100:+.2f},{P[nm]['dacc_range'][1]*100:+.2f}] p {P[nm]['p_dacc_lower']:.3f}" for nm in ctrl))

    # ------------------------------------------------------------------ C
    m0 = mass(x); lm0 = np.log(np.maximum(m0, 1e-3))
    log(f'mass check: corr(four-vector mass from particles, Q mass) {np.corrcoef(m0, Q["mass"])[0, 1]:.4f}')
    eps = 0.05
    def ang(xx, k): e = xx.copy(); e[..., 1:] *= np.float32(k); return e
    def iso(xx, k): e = ang(xx, k); e[..., 0] *= np.float32(1 / k); return e
    def rot(xx, th):
        e = xx.copy(); c_, s_ = np.float32(np.cos(th)), np.float32(np.sin(th))
        e[..., 1] = xx[..., 1] * c_ - xx[..., 2] * s_; e[..., 2] = xx[..., 1] * s_ + xx[..., 2] * c_; return e
    EDS = dict(angle=(ang(x, 1 + eps), ang(x, 1 - eps)), isomass=(iso(x, 1 + eps), iso(x, 1 - eps)), rotation=(rot(x, eps), rot(x, -eps)))
    ok = np.ones(J, bool)
    for a_, b_ in EDS.values(): ok &= in_range(net, a_) & in_range(net, b_)
    dlm = np.log(np.maximum(mass(EDS['angle'][0]), 1e-3)) - np.log(np.maximum(mass(EDS['angle'][1]), 1e-3))
    dlm_iso = np.log(np.maximum(mass(EDS['isomass'][0]), 1e-3)) - np.log(np.maximum(mass(EDS['isomass'][1]), 1e-3))
    dlm_rot = np.log(np.maximum(mass(EDS['rotation'][0]), 1e-3)) - np.log(np.maximum(mass(EDS['rotation'][1]), 1e-3))
    DZ = {nm: net.run(a_, keep=('z',))['z'] - net.run(b_, keep=('z',))['z'] for nm, (a_, b_) in EDS.items()}
    res['C'] = dict(n_in_range=int(ok.sum()), eps=eps, dlnm_angle_median=float(np.median(dlm[ok])), dlnm_isomass_median_abs=float(np.median(np.abs(dlm_iso[ok]))),
                    dlnm_rotation_median_abs=float(np.median(np.abs(dlm_rot[ok]))), bins=MBINS, units={})
    mb = np.digitize(m0, MBINS[1:-1])
    for k in ZU:
        z = B['z'][:, k]; rows = []
        for b in range(len(MBINS) - 1):
            for sgn, msk in (('neg', z < 0), ('pos', z > 0)):
                ii = np.flatnonzero(ok & (mb == b) & msk)
                if len(ii) < 40: continue
                bs = [rng.choice(ii, len(ii)) for _ in range(NB)]
                f = lambda d, i: d[i].mean() / dlm[i].mean()
                sl = {nm: float(f(DZ[nm][:, k], ii)) for nm in DZ}
                sl_ci = {nm: ci([f(DZ[nm][:, k], i) for i in bs]) for nm in DZ}
                pz = float(min(1, 2 * min((np.array([f(DZ['angle'][:, k], i) for i in bs]) <= 0).mean(),
                                          (np.array([f(DZ['angle'][:, k], i) for i in bs]) >= 0).mean()) + 1 / NB))
                diff = [f(DZ['angle'][:, k] - DZ['isomass'][:, k], i) for i in bs]
                rows.append(dict(bin=f'{MBINS[b]}-{MBINS[b + 1]}', sign=sgn, n=int(len(ii)), z_mean=float(z[ii].mean()),
                                 lnm_mean=float(lm0[ii].mean()), dz_dlnm=sl, dz_dlnm_ci=sl_ci, p_angle_zero=pz,
                                 angle_minus_isomass=float(np.mean(diff)), angle_minus_isomass_ci=ci(diff)))
        # descriptive slope: binned mean z (all jets) vs mean ln m, finite differences between adjacent bins
        zm = [float(z[ok & (mb == b)].mean()) for b in range(len(MBINS) - 1)]
        lmm = [float(lm0[ok & (mb == b)].mean()) for b in range(len(MBINS) - 1)]
        desc = [(zm[b + 1] - zm[b]) / (lmm[b + 1] - lmm[b]) for b in range(len(zm) - 1)]
        res['C']['units'][f'z{k}'] = dict(rows=rows, z_mean_by_bin=zm, lnm_mean_by_bin=lmm, descriptive_slope_between_bins=desc)
        log(f'z{k} dz/dlnm (angle | isomass | rotation, per angle-edit Δln m) by mass bin and sign:')
        for rw in rows:
            log(f"   {rw['bin']:>9} {rw['sign']} n {rw['n']:5d} z̄ {rw['z_mean']:+6.2f}: angle {rw['dz_dlnm']['angle']:+7.2f} {[round(v, 2) for v in rw['dz_dlnm_ci']['angle']]} p {rw['p_angle_zero']:.3g} | "
                f"iso {rw['dz_dlnm']['isomass']:+7.2f} | rot {rw['dz_dlnm']['rotation']:+7.2f} | angle−iso {rw['angle_minus_isomass']:+6.2f} {[round(v, 2) for v in rw['angle_minus_isomass_ci']]}")
        log(f'   descriptive slope between bins: {[round(v, 2) for v in desc]}')
    (OUT / 'e30_verify.json').write_text(json.dumps(res, indent=1, default=float))
    log('done')


if __name__ == '__main__':
    main()
