"""E33 verifier (fresh samples; no training).  python -m interp.e33_verify [--models m1,m2] [--smoke]
-> $AI_RESULTS/jedi/e33_verify.json (and e33_verify_<model>.json per model as it finishes)

Fresh samples: fit = sample(2000, 4401), test = sample(2000, 4402) (E33 used 3301 / 3302).  For every trained E33 model:
 A  rerun of E33 step 3 (four role fits, Hungarian role assignment, 250-rotation R^2 null, rng 4405) and step 4 (one-particle
    edits with E33's controls, rng 4406; slope, control ratio, sign-flip p, found / found_dir) with E33's own functions.
 B  missing control for the 'hard' role (sum pT of the k hardest): E33's control nudges a rank >= k particle by x 1.1, i.e. a
    much smaller pT change than the in-window nudge, so |dS_ctrl| < |dS_in| is expected even for a unit that sums ALL pT.
    Here: response per GeV, beta = sum(dS * dpT) / sum(dpT^2), for the in-window x 1.1 nudge and for (i) E33's x 1.1 control and
    (ii) a matched-step control (the in-window dpT added to a random rank >= k particle, capped at the pT of rank - 1).
    H0 (no rank window: unit ~ total pT): beta_ctrl / beta_in = 1;  H1 (window): ratio < 1 (window formula predicts 0).
    Paired jet bootstrap (1,000) for the ratio; p = (1 + #{ratio_b >= 1}) / 1001.  Also R^2 of the hard unit vs total pT (k = n).
 C  competing formulas for the assigned units: r2 unit vs r1 = a + b sum pT r + c sum r + d N(pT<2, r>0.1) and vs multiplicity N;
    soft unit vs N alone; hard unit vs total pT.
 D  decoy-family null for 'recurrence': 30 random hinge families f_q = sum_{rank<k} ReLU(cos th z_pT + sin th z_r + b) (k = 1..n,
    th ~ U(0, 2 pi), b ~ N(0, 1), z standardised over real particles of the fit jets; seed 4407), plus fixed decoys
    sum_{rank<k} r, sum_{rank<k} pT^2, sum_{rank>=k} (r - t)+.  Per model and family: best live S unit by fit R^2, test R^2.
    Recurrence base rate = among decoys with test R^2 >= 0.9 in official_n64, the share of the other 9 trained models where
    they also reach 0.9; compared with the same count for the four roles (best unit, unassigned).
"""
import argparse, gc, json, os, sys, time
from pathlib import Path
import numpy as np

os.environ.setdefault('KERAS_BACKEND', 'jax')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import e33_program_recurrence_jedi as E  # noqa: E402

OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
T0 = time.time()
NDEC = 30


def log(*a): print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


def beta(o, d):
    d2 = (d ** 2).sum(); return float((o * d).sum() / d2) if d2 > 0 else np.nan


def r2_lin(zf, zt, Gf, Gt):
    r2f, r2t, _ = E.fit_set(zf[:, None], zt[:, None], Gf, Gt); return float(r2t[0])


def hinge_feat(xk, fam, zs):
    pt, r, real, _ = E.particles(xk); rf = real.astype(np.float64); th, b = fam
    zp = (pt - zs[0]) / zs[1]; zr = (r - zs[2]) / zs[3]
    return np.cumsum(rf * np.maximum(np.cos(th) * zp + np.sin(th) * zr + b, 0), 1)


def best_unit(Zf, Zt, Ff, Ft):
    r2f, r2t, _, _ = E.single_fit(Zf, Zt, Ff, Ft); i = np.unravel_index(np.nanargmax(np.where(np.isfinite(r2f), r2f, -9)), r2f.shape)
    return float(r2t[i]), int(i[1])


def run_one(name, per, nrot, nboot, fams):
    from scipy.optimize import linear_sum_assignment  # noqa: F401
    kind, fam, n = E.MODELS[name]
    model = E.Model(name); f16 = model.f16
    xf, xkf, _, yf, _ = E.load(per, 4401, n, f16); xt, xkt, _, yt, _ = E.load(per, 4402, n, f16)
    R = dict(model=name, N_fit=len(yf), N_test=len(yt))
    pt = xkt[..., 0]; real = pt > 0
    R['sorted_share'] = float(np.mean([(np.diff(p[m]) <= 1e-6).all() for p, m in zip(pt[:2000], real[:2000])]))
    R['padding_after_real_share'] = float(np.mean([(not m[np.argmin(m):].any()) if (~m).any() else True for m in real[:2000]]))
    St = model.run(xt, keep=('S', 'logits')); R['acc'] = float((St['logits'].argmax(1) == yt).mean())
    Sf = model.run(xf, keep=('S',))['S']; Stt = St['S']
    liveS = [c for c in range(64) if Sf[:, c].std() > 1e-9]; R['n_live_S'] = len(liveS)
    Zf, Zt = Sf[:, liveS], Stt[:, liveS]
    RF = E.role_feats(xkf); RT = E.role_feats(xkt)
    rf = E.roles_fit(Zf, Zt, RF, RT); R['roles'] = {}
    mu, sd = Zf.mean(0), Zf.std(0); sd[sd == 0] = 1; Zfs = (Zf - mu) / sd; Zts = (Zt - mu) / sd
    rng_r = np.random.default_rng(4405); rot = {r: [] for r in E.ROLES}
    for _ in range(nrot):
        Q = E.rand_rot(len(liveS), rng_r); rr = E.roles_fit(Zfs @ Q, Zts @ Q, RF, RT)
        for r in E.ROLES: rot[r].append(rr[r]['r2_test'] if rr[r]['r2_test'] is not None else -1.0)
    pf, ptt = E.particles(xkf), E.particles(xkt)
    Nf, Nt = pf[2].sum(1)[:, None].astype(float), ptt[2].sum(1)[:, None].astype(float)

    def r1(P):
        p_, r_, re_, _ = P; rf_ = re_.astype(float)
        return np.stack([(p_ * r_).sum(1), (rf_ * r_).sum(1), (rf_ * ((p_ < 2) & (r_ > 0.1))).sum(1)], 1)
    rng_e = np.random.default_rng(4406)
    for role in E.ROLES:
        e = rf[role]; u = e['unit_idx']
        out = dict(unit=f'S{liveS[u]}' if u is not None else None, param=e['param'], r2_test=e['r2_test'],
                   rot_rate=float((np.array(rot[role]) >= 0.9).mean()))
        if u is None: R['roles'][role] = out; continue
        c = liveS[u]; zf, zt = Sf[:, c], Stt[:, c]; prm = e['param']; coef = np.array(e['coef'][1:])
        # C: competing formulas
        if role == 'hard': out['r2_total_pT'] = r2_lin(zf, zt, RF['hard'][-1][1], RT['hard'][-1][1])
        if role == 'r2':
            out['r2_r1_alt'] = r2_lin(zf, zt, r1(pf), r1(ptt)); out['r2_mult'] = r2_lin(zf, zt, Nf, Nt)
        if role in ('soft', 'core'): out['r2_mult'] = r2_lin(zf, zt, Nf, Nt)
        # A: E33 edits
        f0 = E.role_value(role, prm, xkt) @ coef; ed = E.make_edits(role, prm, xkt, rng_e); res = {}

        def resp(I, xe):
            xin = E.to16(xt[I], xkt[I], xe) if f16 else xe
            return model.run(xin, keep=('S',))['S'][:, c] - Stt[I, c]
        for nm, (I, xe) in ed.items():
            if len(I) < 10: res[nm] = None; continue
            res[nm] = dict(I=I, obs=resp(I, xe), pred=E.role_value(role, prm, xe) @ coef - f0[I],
                           dpt=(xe[..., 0].astype(np.float64) - xkt[I][..., 0]).sum(1))
        ci = res.get('in')
        if ci is not None:
            s, sci = E.boot_slope(ci['obs'], ci['pred'], rng_e, nboot)
            out['in'] = dict(n=len(ci['I']), slope=s, slope_ci=sci, covers1=bool(sci[0] is not None and sci[0] <= 1 <= sci[1]))
            ctr = {}
            for nm, v in res.items():
                if nm == 'in' or v is None: continue
                Ic, ii, jj = np.intersect1d(ci['I'], v['I'], return_indices=True)
                oc, pin, oin = np.abs(v['obs'][jj]), np.abs(ci['pred'][ii]), np.abs(ci['obs'][ii])
                rc, rci = E.boot_ratio(oc, pin, rng_e, nboot); rin, rinci = E.boot_ratio(oin, pin, rng_e, nboot)
                ctr[nm] = dict(n=len(Ic), ratio_ctrl=rc, ratio_ctrl_ci=rci, ratio_in=rin, ratio_in_ci=rinci, p_signflip=E.signflip_p(oin - oc, rng_e, 10000))
            out['controls'] = ctr
            out['found'] = bool((e['r2_test'] or -1) >= 0.9 and out['in']['covers1'])
            out['found_dir'] = bool((e['r2_test'] or -1) >= 0.9 and sci[0] is not None and sci[0] > 0 and ctr
                                    and all(v['p_signflip'] < 0.05 and (v['ratio_in'] or 0) > (v['ratio_ctrl'] or 0) for v in ctr.values()))
            # B: per-GeV hard control
            if role == 'hard' and prm['k'] < n:
                k = prm['k']; ptk, _, rk, _ = E.particles(xkt); J = len(ptk); rank = np.arange(n)[None]
                prev = np.concatenate([np.full((J, 1), np.inf), ptk[:, :-1]], 1)
                dfull = np.zeros(J); dfull[ci['I']] = ci['dpt']
                jc = E.pick(rk & (rank >= k), rng_e); ok = (jc >= 0) & (dfull > 0); I2 = np.flatnonzero(ok)
                xe = xkt[I2].astype(np.float64).copy(); a = np.arange(len(I2)); j2 = jc[I2]
                xe[a, j2, 0] = np.minimum(ptk[I2, j2] + dfull[I2], prev[I2, j2]); xe = xe.astype(np.float32)
                res['ctrl_matched'] = dict(I=I2, obs=resp(I2, xe), dpt=(xe[..., 0].astype(np.float64) - xkt[I2][..., 0]).sum(1))
                pg = {}
                for nm in ('ctrl_outside', 'ctrl_matched'):
                    v = res.get(nm)
                    if v is None or len(v['I']) < 10: pg[nm] = None; continue
                    Ic, ii, jj = np.intersect1d(ci['I'], v['I'], return_indices=True)
                    oi, di, oc, dc = ci['obs'][ii], ci['dpt'][ii], v['obs'][jj], v['dpt'][jj]
                    bi, bc = beta(oi, di), beta(oc, dc); bs = []
                    for _ in range(nboot):
                        q = rng_e.integers(0, len(Ic), len(Ic)); bs.append(beta(oc[q], dc[q]) / beta(oi[q], di[q]))
                    bs = np.array(bs)
                    pg[nm] = dict(n=len(Ic), beta_in_per_GeV=bi, beta_ctrl_per_GeV=bc, ratio=bc / bi,
                                  ratio_ci=[float(np.nanquantile(bs, .025)), float(np.nanquantile(bs, .975))],
                                  p_H0_no_window=float((1 + np.sum(bs >= 1)) / (1 + nboot)),
                                  mean_dpt_in=float(di.mean()), mean_dpt_ctrl=float(dc.mean()))
                out['per_GeV'] = pg
        R['roles'][role] = out
        log(f'  {name} {role} {out["unit"]} {prm} R2 {out["r2_test"]:.3f} rot {out["rot_rate"]:.3f} '
            f'slope {out.get("in", {}).get("slope")} {out.get("in", {}).get("slope_ci")} dir {out.get("found_dir")} '
            + json.dumps(E.clean({k: v for k, v in out.items() if k in ('r2_total_pT', 'r2_r1_alt', 'r2_mult', 'per_GeV', 'controls')})))
    # D: decoys
    zs = (pf[0][pf[2]].mean(), pf[0][pf[2]].std(), pf[1][pf[2]].mean(), pf[1][pf[2]].std())
    dec = {}
    for q in range(len(fams)): dec[f'rand{q}'] = best_unit(Zf, Zt, hinge_feat(xkf, fams[q], zs), hinge_feat(xkt, fams[q], zs))[0]
    for nm, fn in (('sum_r_prefix', lambda P: np.cumsum(P[1], 1)), ('sum_pT2_prefix', lambda P: np.cumsum(P[0] ** 2, 1))):
        dec[nm] = best_unit(Zf, Zt, fn(pf), fn(ptt))[0]

    def outer(P):
        p_, r_, re_, _ = P; rank = np.arange(r_.shape[1])[None]
        return np.concatenate([(np.maximum(r_ - t, 0) * re_ * (rank >= k)).sum(1)[:, None] for t in (0.05, 0.1, 0.2, 0.3, 0.4) for k in E.K_SOFT if k < n], 1)
    dec['outer_ramp'] = best_unit(Zf, Zt, outer(pf), outer(ptt))[0]
    R['decoys'] = dec
    R['roles_best_unit'] = {r: best_unit(Zf, Zt, np.concatenate([g for _, g in RF[r]], 1), np.concatenate([g for _, g in RT[r]], 1))[0]
                            if r != 'r2' else float(np.nanmax(rf['r2']['per_unit_test'])) for r in E.ROLES}
    log(f'  {name} decoys >= 0.9: {sum(v >= 0.9 for v in dec.values())}/{len(dec)}; roles best unit {R["roles_best_unit"]}')
    del model; gc.collect()
    try:
        import jax; jax.clear_caches()
    except Exception: pass
    return R


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--models', default=','.join(E.TRAINED)); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    per, nrot, nboot = (200, 10, 100) if a.smoke else (2000, 250, 1000)
    rng = np.random.default_rng(4407); fams = [(float(rng.uniform(0, 2 * np.pi)), float(rng.standard_normal())) for _ in range(NDEC)]
    S = dict(node='E33-verify', samples=f'fit sample({per}, 4401), test sample({per}, 4402)', decoy_families=fams, models={})
    tag = 'e33_verify' + ('_smoke' if a.smoke else '')
    for m in a.models.split(','):
        log('model', m); S['models'][m] = run_one(m, per, nrot, nboot, fams)
        (OUT / f'{tag}_{m}.json').write_text(json.dumps(E.clean(S['models'][m]), indent=1))
        (OUT / f'{tag}.json').write_text(json.dumps(E.clean(S), indent=1))
    ms = [m for m in E.TRAINED if m in S['models']]; summ = {}
    for role in E.ROLES:
        nr = [(1 + S['models'][m]['roles'][role]['rot_rate'] * nrot) / (1 + nrot) for m in ms]
        fd = [m for m in ms if S['models'][m]['roles'][role].get('found_dir')]
        fo = [m for m in ms if S['models'][m]['roles'][role].get('found')]
        r9 = [m for m in ms if (S['models'][m]['roles'][role]['r2_test'] or -1) >= 0.9]
        summ[role] = dict(found=fo, found_dir=fd, r2_ok=r9, expected_null=float(sum(nr)), p_dir=E.pbinom_tail(nr, len(fd)), p_r2=E.pbinom_tail(nr, len(r9)))
    if 'official_n64' in S['models']:
        ref = S['models']['official_n64']; others = [m for m in ms if m != 'official_n64']
        dref = [d for d, v in ref['decoys'].items() if v >= 0.9]
        rec = {d: sum(S['models'][m]['decoys'][d] >= 0.9 for m in others) for d in dref}
        rrec = {r: sum(S['models'][m]['roles_best_unit'][r] >= 0.9 for m in others) for r in E.ROLES if ref['roles_best_unit'][r] >= 0.9}
        summ['decoy_null'] = dict(n_decoys=len(ref['decoys']), ref_matched=dref, recurrence_counts=rec,
                                  decoy_rate=float(np.mean(list(rec.values())) / len(others)) if rec else None, role_counts=rrec, n_other=len(others))
    S['summary'] = summ
    (OUT / f'{tag}.json').write_text(json.dumps(E.clean(S), indent=1)); log('summary', json.dumps(E.clean(summ)))


if __name__ == '__main__':
    main()
