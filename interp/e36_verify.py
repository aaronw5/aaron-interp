"""E36 verification (fresh sample, metric sensitivity, direction controls, program-agreement baseline).

1. Rerun of E36's search on a fresh sample sample(300, 3701) minus the E36 main/replicate/pool jets, angle unit SC = 0.02 rad
   (as E36) and SC = 0.05 rad (angular moves 6.25x cheaper in eps^2): transfer to the exact network, random same-v-norm edits,
   coordinate / rank shares, pT^D, M2, hardest-particle pT change, S20/S51/S43/S0/S1/S29/S30/S19 |dS|/sd.
2. Direction controls on the found edits (exact network, all re-projected onto the constraints with 30 Gauss-Newton steps):
   sign-randomised edit (same |v| per entry, random signs: same coordinate/rank profile, random direction), negated edit (-v),
   pT part only (angles zeroed), angle part only (pT zeroed).
3. Hand-made hardness edit at fixed jet four-momentum: leading particle pT x1.5 (or x1/1.5), all other coordinates free in the
   minimum-norm projection; matched control: the same absolute pT given to the rank-5 particle instead.
4. Program agreement against the constant 'never flips' predictor and Cohen's kappa, from E36's stored 2x2 counts.
Writes $AI_RESULTS/jedi/e36_verify.json.   Run: python -m interp.e36_verify [--per 300]
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
os.environ.setdefault('KERAS_BACKEND', 'jax')
from interp import e36_counterfactual_flip_jedi as E  # noqa: E402

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'
T0 = time.time()
UNITS = [20, 51, 43, 0, 1, 29, 30, 19]
HARD_D = (0.5, 1.0, -1 / 3)


def log(*a): print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def proj_free(x0, v, P0, free, iters=30, ramp=10):
    """as E.project, but only the coordinates in `free` (J, n, 3) bool may move; the fixed coordinates' target values are
    approached in `ramp` equal increments with projection after each (Gauss-Newton overshoots on large one-step changes).
    Jets whose iteration goes non-finite are returned with v = 0 (they fail the constraint check only if P0 is off; flagged by
    the returned mask)."""
    real = (x0[..., 0] > 0)[..., None]; fr = (free & real).reshape(len(x0), 1, -1).astype(float)
    tgt = np.where(real & ~free, v, 0.0); v = np.zeros_like(x0); s = np.maximum(np.hypot(P0[:, 0], P0[:, 1]), 1e-9)[:, None]
    bad = np.zeros(len(x0), bool)
    for k in range(1, ramp + 1):
        v = np.where(free, v, tgt * k / ramp)
        for _ in range(iters):
            with np.errstate(all='ignore'):
                r = (E.fourmom(x0, v) - P0) / s; A = E.jac(x0, v) / s[..., None] * fr
                AA = A @ np.transpose(A, (0, 2, 1)) + 1e-12 * np.eye(4)
                fin = np.isfinite(AA).all((1, 2)) & np.isfinite(r).all(1); bad |= ~fin
                AA[~fin] = np.eye(4); r[~fin] = 0; A[~fin] = 0
                d = -np.einsum('jkn,jk->jn', A, np.einsum('jkl,jl->jk', np.linalg.pinv(AA, rcond=1e-13), r))
                d[~np.isfinite(d).all(1)] = 0
            v = v + d.reshape(v.shape); bad |= ~np.isfinite(v).all((1, 2)); v[bad] = 0
    return np.where(real, v, 0.0), ~bad


def gn(x0, v, P0, iters=10):
    """Gauss-Newton minimum-norm projection (all coordinates free) with pseudo-inverse and non-finite guard -> v, good"""
    real = (x0[..., 0] > 0)[..., None]; s = np.maximum(np.hypot(P0[:, 0], P0[:, 1]), 1e-9)[:, None]; good = np.ones(len(x0), bool)
    for _ in range(iters):
        with np.errstate(all='ignore'):
            r = (E.fourmom(x0, v) - P0) / s; A = E.jac(x0, v) / s[..., None]
            A = A * np.repeat(real, 3, -1).reshape(len(x0), 1, -1)
            AA = A @ np.transpose(A, (0, 2, 1)); fin = np.isfinite(AA).all((1, 2)) & np.isfinite(r).all(1); good &= fin
            AA[~fin] = np.eye(4); r[~fin] = 0; A[~fin] = 0
            d = -np.einsum('jkn,jk->jn', A, np.einsum('jkl,jl->jk', np.linalg.pinv(AA, rcond=1e-13), r))
        v = v + d.reshape(v.shape); good &= np.isfinite(v).all((1, 2)); v[~good] = 0
    return np.where(real, v, 0.0), good


def random_ctrl(x0, v_ref, R, rng, ramp=10):
    """R random edits per jet at v_ref's v-space norm, tangent at 0, applied in `ramp` increments with projection after each"""
    P0 = E.p4(x0).sum(1); nv = np.sqrt((v_ref ** 2).sum((1, 2))); out = []
    for _ in range(R):
        d = E.tangent(x0, np.zeros_like(x0), rng.normal(size=x0.shape)); d = d * (nv / np.maximum(np.sqrt((d ** 2).sum((1, 2))), 1e-12))[:, None, None]
        v = np.zeros_like(x0); good = np.ones(len(x0), bool)
        for _k in range(ramp):
            v, g_ = gn(x0, v + d / ramp, P0); good &= g_
        out.append((v, good))
    return out


def calls(net, x0, v):
    xe, _ = E.apply_edit(x0, v); dm, dax, _ = E.check_constraints(x0, xe); r = net.run(xe.astype(np.float32), keep=('logits', 'S'))
    return r['logits'], r['S'], xe, (dm < 1e-3) & (dax < 1e-3)


def one_sc(sc, net, f, x, y, a, rng, mcnemar):
    import jax.numpy as jnp
    from interp.e31_onmanifold_jedi import jet_feats
    E.SC = sc; step = E.make_step(f, a.kappa)
    x0 = np.asarray(x, np.float64); J = len(x0); out = dict(SC=sc, N=J)
    ex0 = net.run(x, keep=('logits', 'S')); c_ex = ex0['logits'].argmax(1)
    lo_f = np.concatenate([np.asarray(f(jnp.asarray(x[i:i + 1000]))) for i in range(0, J, 1000)]); c_f = lo_f.argmax(1)
    tgt = np.array([E.PARTNER.get(int(c), -1) for c in c_f]); tgt = np.where(tgt < 0, np.argsort(-lo_f, 1)[:, 1], tgt)
    sel = c_f == c_ex; out.update(acc_exact=float((c_ex == y).mean()), same_call=float(sel.mean()))
    v, found, _ = E.search(step, x0, tgt, a.steps, a.lr, tag=f'SC {sc}')
    lo_e, S_e, xe, cm = calls(net, x0, v); c_e = lo_e.argmax(1)
    ok = sel & found & cm; prim = ok & (c_f != E.T_); eps = E.eps_of(x0, v)
    grp = {'W->Z': c_f == E.W_, 'Z->W': c_f == E.Z_, 'q->g': c_f == E.Q_, 'g->q': c_f == E.G_}
    tr = (c_e == tgt).astype(float)
    out['found_float'] = float(found[sel & (c_f != E.T_)].mean()); out['n_primary_ok'] = int(prim.sum())
    out['eps_median'] = float(np.median(eps[prim])); out['transfer'] = E.boot_mean(tr[prim])
    out['transfer_by_group'] = {g: E.boot_mean(tr[prim & m]) for g, m in grp.items()}
    rv_, rf_ = E.search.robust; lo_r, _, _, cmr = calls(net, x0, rv_); rok = sel & rf_ & cmr & (c_f != E.T_)
    out['robust_transfer'] = E.boot_mean((lo_r.argmax(1) == tgt)[rok].astype(float))
    log(f'SC {sc}: N {J}, primary ok {prim.sum()}, transfer {out["transfer"]["mean"]:.3f} {out["transfer"]["ci"]}, median eps {out["eps_median"]:.2f}')
    idx = np.flatnonzero(prim); x0i, vi, ti = x0[idx], v[idx], tgt[idx]; P0 = E.p4(x0i).sum(1)
    # random same-v-norm edits
    R = a.R; rfl = np.full((len(idx), R), np.nan); RV0 = None; Sr0 = None
    for r, vr in enumerate(E.random_edits(x0i, vi, R, rng)):
        lo_, S_, _, c_ = calls(net, x0i, vr); rfl[:, r] = np.where(c_, lo_.argmax(1) == ti, np.nan)
        if r == 0: RV0, Sr0 = vr, S_
    rmean = np.nanmean(rfl, 1); out['random_flip'] = E.boot_mean(rmean)
    out['found_minus_random'] = E.boot_diff(tr[idx], rmean)
    out['mcnemar_found_vs_random_draw0'] = mcnemar(tr[idx].astype(bool), rfl[:, 0] == 1)
    # direction controls
    real = x0i[..., 0] > 0; ctl = {}
    sgn = rng.choice([-1.0, 1.0], size=vi.shape)
    ctl_v = {'sign-randomised (same |v| profile)': E.project(x0i, vi * sgn, P0, iters=30), 'negated (-v)': E.project(x0i, -vi, P0, iters=30),
             'pT part only': E.project(x0i, vi * np.array([1.0, 0, 0]), P0, iters=30), 'angle part only': E.project(x0i, vi * np.array([0, 1.0, 1.0]), P0, iters=30)}
    lo0 = ex0['logits'][idx]; cur = c_ex[idx]
    def marg(lo): return lo[np.arange(len(lo)), ti] - lo[np.arange(len(lo)), cur]   # target minus current-call logit
    out['margin_unedited_mean'] = float(marg(lo0).mean()); out['margin_found'] = E.boot_mean(marg(lo_e[idx]) - marg(lo0))
    for k, vv in ctl_v.items():
        lo_, _, _, c_ = calls(net, x0i, vv); fl = (lo_.argmax(1) == ti) & c_
        ctl[k] = dict(flip=E.boot_mean(fl.astype(float)), compliant=float(c_.mean()), eps_over_found_median=float(np.median(E.eps_of(x0i, vv) / eps[idx])),
                      d_margin=E.boot_mean(marg(lo_) - marg(lo0)), found_minus_ctrl=E.boot_diff(tr[idx], fl.astype(float)),
                      mcnemar=mcnemar(tr[idx].astype(bool), fl))
        log(f'SC {sc}: control {k}: flip {ctl[k]["flip"]["mean"]:.3f}, d margin {ctl[k]["d_margin"]["mean"]:.3f}, eps ratio {ctl[k]["eps_over_found_median"]:.2f}')
    out['direction_controls'] = ctl
    # shares, found vs random draw 0
    def shares(xx, vv):
        Pp = E.eps2_parts(xx, vv); tot = sum(Pp.values()).sum(1) + 1e-30; allc = sum(Pp.values()); rk = np.arange(64)[None]
        return {'pT': Pp['pt'].sum(1) / tot, 'radial': Pp['rad'].sum(1) / tot, 'azimuthal': Pp['az'].sum(1) / tot,
                'ranks 0-13': (allc * (rk < 14)).sum(1) / tot, 'rank 0': allc[:, 0] / tot}
    sf, sr = shares(x0i, vi), shares(x0i, RV0)
    out['shares'] = {k: dict(found=E.boot_mean(sf[k]), random=E.boot_mean(sr[k]), diff=E.boot_diff(sf[k], sr[k])) for k in sf}
    out['share_by_group_found'] = {g: {k: float(sf[k][m[idx]].mean()) for k in sf} for g, m in grp.items() if m[idx].sum() >= 20}
    # observables
    F0, FE, FR, FA = jet_feats(x0i), jet_feats(xe[idx]), jet_feats(E.apply_edit(x0i, RV0)[0]), jet_feats(x0)
    out['obs_delta_sd'] = {k: dict(found=E.boot_mean((FE[k] - F0[k]) / FA[k].std()), random=E.boot_mean((FR[k] - F0[k]) / FA[k].std()),
                                   by_group={g: float(((FE[k] - F0[k]) / FA[k].std())[m[idx]].mean()) for g, m in grp.items() if m[idx].sum() >= 20})
                           for k in ('ptD', 'M2', 'tau21', 'D2', 'girth')}
    out['hardest_particle_pT_change_median'] = dict(found=float(np.median(np.expm1(vi[:, 0, 0]))), random=float(np.median(np.expm1(RV0[:, 0, 0]))))
    out['sum_abs_dpT_over_jet_pT_median'] = dict(found=float(np.median((np.abs(np.expm1(vi[..., 0])) * x0i[..., 0]).sum(1) / F0['jpt'])),
                                                 random=float(np.median((np.abs(np.expm1(RV0[..., 0])) * x0i[..., 0]).sum(1) / F0['jpt'])))
    sdS = ex0['S'].std(0) + 1e-9
    out['S_abs_dS_sd'] = {f'S{u}': dict(found=float(np.abs((S_e[idx, u] - ex0['S'][idx, u]) / sdS[u]).mean()),
                                        random=float(np.abs((Sr0[:, u] - ex0['S'][idx, u]) / sdS[u]).mean())) for u in UNITS}
    log(f'SC {sc}: shares found/random ' + ', '.join(f'{k} {out["shares"][k]["found"]["mean"]:.3f}/{out["shares"][k]["random"]["mean"]:.3f}' for k in sf))
    log(f'SC {sc}: ptD found {out["obs_delta_sd"]["ptD"]["found"]["mean"]:.3f} random {out["obs_delta_sd"]["ptD"]["random"]["mean"]:.3f}; '
        f'hardest dpT {out["hardest_particle_pT_change_median"]}; S {out["S_abs_dS_sd"]}')
    return out, dict(x0=x0, c_ex=c_ex, c_f=c_f, sel=sel, lo0=ex0['logits'])


def hardness(net, k, rng, mcnemar):
    """hand-made edit: leading particle pT x(1+d), other coordinates free (min-norm projection); control: same absolute pT to rank 5"""
    E.SC = 0.02
    x0, c_ex, c_f, sel, lo0 = k['x0'], k['c_ex'], k['c_f'], k['sel'], k['lo0']
    nreal = (x0[..., 0] > 0).sum(1); use = sel & (c_ex != E.T_) & (nreal >= 8); idx = np.flatnonzero(use); xi = x0[idx]; P0 = E.p4(xi).sum(1)
    part = np.array([E.PARTNER[int(c)] for c in c_ex[idx]]); cur = c_ex[idx]
    def marg(lo): return lo[np.arange(len(lo)), part] - lo[np.arange(len(lo)), cur]
    m0 = marg(lo0[idx]); out = dict(N=int(len(idx)), definition='margin = logit(partner class) - logit(current call); partner W<->Z, q<->g')
    grp = {'W': cur == E.W_, 'Z': cur == E.Z_, 'q': cur == E.Q_, 'g': cur == E.G_}
    from interp.e31_onmanifold_jedi import jet_feats
    F0 = jet_feats(xi); sdp = F0['ptD'].std()
    for d in HARD_D:
        res = {}
        for lab, slot in (('leading particle', 0), ('rank-5 particle (same absolute pT)', 5)):
            v = np.zeros_like(xi); dpt = d * xi[:, 0, 0]; v[:, slot, 0] = np.log1p(dpt / xi[:, slot, 0]) if slot else np.log1p(d)
            okd = np.isfinite(v[:, slot, 0])
            v[~okd] = 0
            free = np.ones_like(xi, bool); free[:, slot, :] = False
            vv, good = proj_free(xi, v, P0, free); okd &= good
            lo_, _, xe, cm = calls(net, xi, vv); fl = (lo_.argmax(1) == part) & cm & okd
            dmg = marg(lo_) - m0; FE = jet_feats(xe)
            res[lab] = dict(compliant=float((cm & okd).mean()), flip_to_partner=E.boot_mean(fl.astype(float)), d_margin=E.boot_mean(dmg[cm & okd]),
                            by_group={g: dict(flip=float(fl[m].mean()), d_margin=float(dmg[m & cm & okd].mean()), n=int(m.sum())) for g, m in grp.items()},
                            d_ptD_sd=float(np.mean((FE['ptD'] - F0['ptD'])[cm & okd]) / sdp), eps_median=float(np.median(E.eps_of(xi, vv)[okd])), _fl=fl, _v=vv, _ok=cm & okd)
            log(f'hardness d {d:+.2f} {lab}: flip {fl.mean():.3f}, d margin {res[lab]["d_margin"]["mean"]:.3f}, by group '
                + ', '.join(f'{g} {r["flip"]:.3f}/{r["d_margin"]:.3f}' for g, r in res[lab]['by_group'].items()) + f'; ptD {res[lab]["d_ptD_sd"]:.2f} sd; eps {res[lab]["eps_median"]:.2f}')
        a_, b_ = res['leading particle'], res['rank-5 particle (same absolute pT)']
        both = a_['_ok'] & b_['_ok']; res['n_both_defined'] = int(both.sum())
        res['leading_minus_rank5_flip'] = E.boot_diff(a_['_fl'][both], b_['_fl'][both]); res['mcnemar_leading_vs_rank5'] = mcnemar(a_['_fl'][both], b_['_fl'][both])
        # random same-v-norm control for the leading edit
        rr = np.zeros(len(idx)); nn = np.zeros(len(idx)); rdm = np.zeros(len(idx)); R = 10
        for vr, gd in random_ctrl(xi, a_['_v'], R, rng):
            lo_, _, _, cm = calls(net, xi, vr); c2 = cm & gd; rr += ((lo_.argmax(1) == part) & c2); nn += c2; rdm += np.where(c2, marg(lo_) - m0, 0)
        hv = (nn > 0) & a_['_ok']; rr = rr[hv] / nn[hv]; rdm = rdm[hv] / nn[hv]
        res['random_same_norm'] = dict(n=int(hv.sum()), flip=E.boot_mean(rr), d_margin=E.boot_mean(rdm))
        res['leading_minus_random_flip'] = E.boot_diff(a_['_fl'][hv].astype(float), rr)
        dml = marg(calls(net, xi, a_['_v'])[0]) - m0
        res['leading_minus_random_abs_d_margin'] = E.boot_diff(np.abs(dml[hv]), np.abs(rdm))
        log(f'hardness d {d:+.2f}: random same-norm flip {rr.mean():.3f}, d margin {rdm.mean():.3f} (n {hv.sum()})')
        for kk in ('leading particle', 'rank-5 particle (same absolute pT)'): [res[kk].pop(z) for z in ('_fl', '_v', '_ok')]
        out[f'leading pT x{1 + d:.3f}'] = res
    return out


def program_baseline(R36):
    """constant 'never flips' predictor and Cohen's kappa from the stored recall/specificity (exact 2x2 reconstruction)"""
    rng = np.random.default_rng(3702); out = {}
    for smp in ('main', 'replicate'):
        P = R36[smp]['programs']; o = {}
        for name, p in P.items():
            if name.startswith('_'): continue
            nF, nN = p['recall_on_net_flips']['n'], p['specificity_on_net_nonflips']['n']
            TP = int(round(p['recall_on_net_flips']['mean'] * nF)); FN = nF - TP; TN = int(round(p['specificity_on_net_nonflips']['mean'] * nN)); FP = nN - TN
            lab = np.repeat([0, 1, 2, 3], [TP, FN, FP, TN]); N = len(lab)
            def stats(l):
                tp, fn, fp, tn = [(l == c).sum() for c in range(4)]; n = len(l); po = (tp + tn) / n; pn, pp = (tp + fn) / n, (tp + fp) / n
                pe = pn * pp + (1 - pn) * (1 - pp); return po, (fp + tn) / n, (po - pe) / (1 - pe), 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
            po, _, kap, bal = stats(lab); const = (FP + TN) / N
            bs = np.array([stats(lab[rng.integers(0, N, N)]) for _ in range(1000)])
            bsc = bs[:, 1]
            # paired agreement program vs constant: program right & const wrong = TP; const right & program wrong = FP
            from interp.e28w_split_route_jedi import binom_p
            o[name] = dict(table=dict(TP=TP, FN=FN, FP=FP, TN=TN), agreement=float(po), agreement_check=p['agreement_found']['mean'],
                           constant_never_flips=dict(mean=float(const), ci=E.qci(bsc)), program_minus_constant=float(po - const),
                           mcnemar_program_vs_constant=dict(program_only_right=TP, constant_only_right=FP, p=float(binom_p(TP, TP + FP))),
                           cohen_kappa=dict(mean=float(kap), ci=E.qci(bs[:, 2])), balanced_accuracy=dict(mean=float(bal), ci=E.qci(bs[:, 3])))
            log(f'{smp} {name}: agreement {po:.4f} (stored {p["agreement_found"]["mean"]:.4f}), never-flip {const:.4f}, diff {po - const:+.4f}, '
                f'McNemar {TP}:{FP} p {o[name]["mcnemar_program_vs_constant"]["p"]:.3g}, kappa {kap:.3f} {o[name]["cohen_kappa"]["ci"]}, balanced {bal:.3f}')
        out[smp] = o
    return out


def main(a):
    from interp.jedi_core import Jedi, sample
    from interp.e11_mars_all import indices
    from interp.e28w_split_route_jedi import mcnemar
    from interp.x10_das_jedi import clean
    import jax
    OUT.mkdir(parents=True, exist_ok=True); JS = OUT / f'e36_verify{"_smoke" if a.smoke else ""}.json'
    V = dict(node='E36 verify', sample=f'sample({a.per}, 3701) minus E36 main/replicate/pool', settings=vars(a))
    save = lambda: JS.write_text(json.dumps(clean(V), indent=1))
    if not a.only_hardness: V['program_baseline'] = program_baseline(json.loads((OUT / 'e36.json').read_text())); save()
    x, y, _ = sample(a.per, 3701); i0 = indices(a.per, 3701)
    used = np.concatenate([indices(1000, 3601), indices(1000, 3602), indices(2000, 3603)]); keep = ~np.isin(i0, used)
    x, y = np.asarray(x[keep]), y[keep]; V['N'] = int(len(y)); V['overlaps_removed'] = int((~keep).sum())
    log(f'fresh jets {len(y)} ({(~keep).sum()} overlaps with E36 samples removed)')
    net = Jedi(64); f = jax.jit(E.float_program(net)); rng = np.random.default_rng(3703)
    if a.only_hardness:
        JS = OUT / 'e36_verify_hardness.json'; V = dict(node='E36 verify, part 2 (hardness edit, +83 % check)', sample=V['sample'], N=V['N'])
        save = lambda: JS.write_text(json.dumps(clean(V), indent=1))
        D = np.load(OUT / 'e36_edits.npz'); pm = D['prim'] & D['ok']; v36 = D['v'].astype(np.float64)[pm]; x36 = D['x0'].astype(np.float64)[pm]
        xe36, _ = E.apply_edit(x36, v36)
        V['e36_hardest_particle_check'] = dict(n=int(pm.sum()), median_dpT_rel_unedited_rank0=float(np.median(np.expm1(v36[:, 0, 0]))),
                                               median_leading_pT_after_over_before=float(np.median(xe36[:, 0, 0] / x36[:, 0, 0]) - 1),
                                               mean_dpT_rel_unedited_rank0=float(np.mean(np.expm1(v36[:, 0, 0]))),
                                               share_rank0_gains=float((v36[:, 0, 0] > 0).mean()))
        log(f"E36 hardest-particle check: {V['e36_hardest_particle_check']}"); save()
        E.SC = 0.02; x0 = np.asarray(x, np.float64); c_ex = net.run(x)['logits']; lo0 = c_ex; c_ex = c_ex.argmax(1)
        import jax.numpy as jnp
        c_f = np.concatenate([np.asarray(f(jnp.asarray(x[i:i + 1000]))) for i in range(0, len(x), 1000)]).argmax(1)
        V['hardness_edit'] = hardness(net, dict(x0=x0, c_ex=c_ex, c_f=c_f, sel=c_f == c_ex, lo0=lo0), rng, mcnemar); save()
        log(f'wrote {JS}'); log('done'); return
    V['by_SC'] = {}; K = None
    for sc in a.sc:
        V['by_SC'][str(sc)], k = one_sc(sc, net, f, x, y, a, rng, mcnemar); save()
        if sc == 0.02: K = k
    if K is not None:
        V['hardness_edit'] = hardness(net, K, rng, mcnemar); save()
    log(f'wrote {JS}'); log('done')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--per', type=int, default=300); ap.add_argument('--steps', type=int, default=200); ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--kappa', type=float, default=0.05); ap.add_argument('--R', type=int, default=20); ap.add_argument('--sc', type=float, nargs='+', default=[0.02, 0.05])
    ap.add_argument('--smoke', type=int, default=0); ap.add_argument('--only-hardness', action='store_true')
    a = ap.parse_args()
    if a.smoke: a.per, a.steps, a.R = a.smoke, 20, 3
    main(a)
