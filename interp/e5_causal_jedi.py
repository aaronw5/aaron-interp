"""E5 step 3: causal checks of the named second-round objects (the top units by cost from e4_e5.json).

Jets: sample(2000 per class, seed 1), i.e. jets not used for the fits. Each edit changes one named quantity; the network's
measured change of S_c (g recomputed, and g frozen at the unedited jet's g) is compared with the change each named
formula predicts on the same edited input:
  hinge   Σ over the unit's pooled slots of ReLU(a·pT or a·ln pT + b·r + c) + f_empty per empty slot, / 16  (mean context)
  H3, H1  the best 3-term / single hand-built linear fit (e4_e5.json), recomputed on the edited particles
Edits (particles stay pT-ordered; edited jets whose inputs leave a slot's fixed-point range are dropped):
  r_cross / r_ctrl   one particle (slot >= 1) moved radially (same azimuth, same pT) to 0.08 beyond / 0.04 inside the
                     unit's hinge r boundary for its pT; the particle starts >= 0.06 inside the boundary
  r_random           one random particle (slot >= 1) moved to r ~ U(0, 0.35)
  pt_transfer        δ = 10% of the leading pT moved from the leading particle to the particle in slot k ∈ [2, 13]
  pt_add             the same δ added to slot k ∈ [2, 13], no compensation
  scale_all          every pT × 1.10
  add_core / add_wide  one 1 GeV particle added at r = 0.05 / 0.32 (random azimuth) to jets with < 64 particles
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.jedi_core import Jedi, sample
from interp.e4_e5_jedi import handbuilt, hinge_sum, hinge_eval, run_pj, masks, OUT

SEED = 1


def in_range(net, x):
    q = net.L['l1'].iq.quantizer
    k_, i_, f_ = (np.asarray(getattr(q, a)).reshape(64, 3) for a in ('k', 'i', 'f'))
    hi = 2.0 ** i_ - 2.0 ** -f_; lo = np.where(k_ > 0, -2.0 ** i_, 0.0)
    xn = net.inputs(x, 64); real = x[..., 0] > 0
    return ~((((xn > hi[None]) | (xn < lo[None])) & real[..., None]).any((1, 2)))


def seen(net, x):
    """the (pT, Δη, Δφ) the network sees after its per-slot input quantizer, in GeV / rad; empty slots stay 0"""
    st = json.loads((Path(__import__('jetdistill.config', fromlist=['DATA']).DATA) / 'norm_stats.json').read_text())['64']
    xq = np.asarray(net.L['l1'].iq(net.inputs(x, 64), training=False), np.float64) * np.array(st['scale']) + np.array(st['shift'])
    return np.where((x[..., :1] > 0), xq, 0.0)


def hinge_sum_seen(net, u, keep, x):
    xq = seen(net, x); real = x[..., 0] > 0
    v = np.where(real, hinge_eval(u['hinge'], np.maximum(xq[..., 0], 1e-3), np.hypot(xq[..., 1], xq[..., 2])), u['f_pad']) * keep[None, :]
    return v.sum(1) / 16


def resort(x):
    o = np.argsort(-x[..., 0], axis=1, kind='stable')
    return np.take_along_axis(x, o[..., None], axis=1)


def lin_formula(x, F, keep):
    H = handbuilt(x)
    if any(t.startswith('K:') for t in F['terms']): H.update(handbuilt(x, keep, prefix='K:'))
    return F['intercept'] + sum(c * H[t] for t, c in zip(F['terms'], F['coef']))


def r_boundary(p, pt):
    u = np.log(np.maximum(pt, 1e-3)) if p['form'] == 'ln' else pt
    return -(p['c'] + p['a'] * u) / p['b']


def move_radial(x, j, rnew):
    xe = x.copy(); J = np.arange(len(x))
    th = np.arctan2(x[J, j, 2], x[J, j, 1])
    xe[J, j, 1] = rnew * np.cos(th); xe[J, j, 2] = rnew * np.sin(th)
    return xe


def main():
    t0 = time.time()
    R5 = json.loads((OUT / 'e4_e5.json').read_text())
    U = [u for u in R5['E5']['units'] if not u.get('context_only')]
    U.sort(key=lambda u: -u['cost']); U = U[:int(sys.argv[2]) if len(sys.argv) > 2 else 3]
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    x, y, Q = sample(per, seed=SEED); J = len(x)
    net = Jedi(64); _, KEEP = masks(net)
    rng = np.random.default_rng(SEED)
    B = net.run(x, keep=('g', 'S', 'logits')); call = B['logits'].argmax(1)
    pt = x[..., 0]; real = pt > 0; n = real.sum(1); r = np.hypot(x[..., 1], x[..., 2])
    print(f'{J} jets ({per}/class, seed {SEED}); accuracy {(call == y).mean():.4f}; units {[u["S"] for u in U]}', flush=True)
    out = dict(N=J, seed=SEED, units=[u['S'] for u in U], edits={})

    def measure(name, xe, valid, unit=None, info=None):
        """valid jets only; returns per unit the measured and predicted ΔS"""
        valid = valid & in_range(net, xe)
        xs, xe_ = x[valid], xe[valid]
        E = net.run(xe_, keep=('S', 'logits')); Fz = run_pj(net, xe_, B['g'][valid], keep=('S',))
        rec = dict(n=int(valid.sum()), calls_changed=float((E['logits'].argmax(1) != call[valid]).mean()), units={})
        for u in ([unit] if unit is not None else U):
            c = u['S']; keep = np.array([s in set(u['pooled_slots']) for s in range(64)])
            dm = E['S'][:, c] - B['S'][valid, c]; dz = Fz['S'][:, c] - B['S'][valid, c]
            ph = hinge_sum(u['hinge'], u['f_pad'], keep, xe_) - hinge_sum(u['hinge'], u['f_pad'], keep, xs)
            p3 = lin_formula(xe_, u['H_greedy3'], keep) - lin_formula(xs, u['H_greedy3'], keep)
            p1 = lin_formula(xe_, u['H_single'], keep) - lin_formula(xs, u['H_single'], keep)
            pq = hinge_sum_seen(net, u, keep, xe_) - hinge_sum_seen(net, u, keep, xs)
            sdS = float(B['S'][:, c].std())

            def cmp(p):
                ok = np.std(p) > 0.05 * max(abs(p.mean()), 1e-9) and np.std(dm) > 1e-9      # corr only when predictions vary
                return dict(mean=float(p.mean()), corr=float(np.corrcoef(p, dm)[0, 1]) if ok else None,
                            slope=float(np.polyfit(p, dm, 1)[0]) if ok else None)
            rec['units'][c] = dict(sd_S=sdS, measured=dict(mean=float(dm.mean()), sem=float(dm.std() / np.sqrt(len(dm))), sd=float(dm.std())),
                                   measured_g_frozen=dict(mean=float(dz.mean()), sem=float(dz.std() / np.sqrt(len(dz))),
                                                          corr_with_full=float(np.corrcoef(dz, dm)[0, 1]) if np.std(dz) > 1e-9 and np.std(dm) > 1e-9 else None),
                                   hinge=cmp(ph), hinge_seen=cmp(pq), H3=cmp(p3), H1=cmp(p1),
                                   H3_terms=u['H_greedy3']['terms'], H1_terms=u['H_single']['terms'])
            d = rec['units'][c]
            f = lambda v: 'n/a' if v is None else f'{v:+.2f}'
            print(f"  {name:12s} S{c:<2d} n={rec['n']:5d} | measured ΔS {d['measured']['mean']:+.3f}±{d['measured']['sem']:.3f} "
                  f"(g frozen {d['measured_g_frozen']['mean']:+.3f}) | predicted hinge {d['hinge']['mean']:+.3f} (corr {f(d['hinge']['corr'])}, slope {f(d['hinge']['slope'])}) "
                  f"hinge on seen input {d['hinge_seen']['mean']:+.3f} (corr {f(d['hinge_seen']['corr'])}) "
                  f"H3 {d['H3']['mean']:+.3f} (corr {f(d['H3']['corr'])}) H1 {d['H1']['mean']:+.3f} (corr {f(d['H1']['corr'])}) | sd(S) {sdS:.2f} | calls changed {rec['calls_changed']*100:.2f}%", flush=True)
        out['edits'].setdefault(name, {}).update({str(k): v for k, v in rec['units'].items()})
        out['edits'][name]['_n'] = rec['n']; out['edits'][name]['_calls_changed'] = rec['calls_changed']
        if info: out['edits'][name]['_info'] = info

    # ---------- targeted r-boundary crossings, one per unit with an r-dependent hinge
    for u in U:
        p = u['hinge']; c = u['S']
        if abs(p['b']) < 1e-6:
            print(f'S{c}: hinge has no r term; r crossings not applicable (r_random below is the control)', flush=True); continue
        rb = r_boundary(p, pt)
        keep = np.array([s in set(u['pooled_slots']) for s in range(64)])
        side = np.sign(p['b'])                                    # +: region is outside the boundary, -: inside
        ok = real & keep[None, :] & (np.arange(64)[None, :] >= 1) & (rb > 0.10) & (rb < 0.30)
        ok &= (r < rb - 0.06) if side > 0 else (r > rb + 0.06)
        has = ok.any(1)
        if has.sum() < 100:
            print(f'S{c}: only {int(has.sum())} jets have a particle inside the r-boundary window; r crossings skipped', flush=True); continue
        j = np.array([rng.choice(np.flatnonzero(o)) if o.any() else 0 for o in ok])
        rbj = rb[np.arange(J), j]
        xc = move_radial(x, j, rbj + 0.08 * side); xk = move_radial(x, j, rbj - 0.04 * side)
        info = dict(r_old_mean=float(r[np.arange(J), j][has].mean()), r_boundary_mean=float(rbj[has].mean()),
                    pt_mean=float(pt[np.arange(J), j][has].mean()), slot_mean=float(j[has].mean()))
        print(f"S{c} r-boundary edit: particle pT {info['pt_mean']:.1f} GeV (mean), slot {info['slot_mean']:.1f}, r {info['r_old_mean']:.3f} -> "
              f"boundary {info['r_boundary_mean']:.3f} {'+' if side > 0 else '-'} 0.08 (cross) or {'-' if side > 0 else '+'} 0.04 (control)", flush=True)
        measure(f'r_cross_S{c}', xc, has, unit=u, info=info)
        measure(f'r_ctrl_S{c}', xk, has, unit=u, info=info)

    # ---------- random radial move (all units)
    cand = real & (np.arange(64)[None, :] >= 1)
    j = np.array([rng.choice(np.flatnonzero(o)) if o.any() else 0 for o in cand])
    rnew = rng.uniform(0, 0.35, J)
    measure('r_random', move_radial(x, j, rnew), cand.any(1))

    # ---------- pT edits
    k = rng.integers(2, 14, J); valid = n > 14
    d = 0.10 * pt[:, 0]
    xt = x.copy(); xt[np.arange(J), k, 0] += d; xt[:, 0, 0] -= d; xt = resort(xt)
    measure('pt_transfer', xt, valid)
    xa = x.copy(); xa[np.arange(J), k, 0] += d; xa = resort(xa)
    measure('pt_add', xa, valid)
    xs = x.copy(); xs[..., 0] *= 1.10
    measure('scale_all', xs, np.ones(J, bool))

    # ---------- add one soft particle
    for nm, rr in (('add_core', 0.05), ('add_wide', 0.32)):
        th = rng.uniform(-np.pi, np.pi, J)
        xn = x.copy(); idx = np.minimum(n, 63)
        xn[np.arange(J), idx] = np.stack([np.full(J, 1.0), rr * np.cos(th), rr * np.sin(th)], 1)
        xn = resort(xn)
        measure(nm, xn, n < 64)

    (OUT / 'e5_causal.json').write_text(json.dumps(out))
    print(f'done {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
