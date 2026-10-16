"""E4–E5 (JEDI-linear 64 p, 3-feature): the context round (claim A2) and the second-round objects (claim A3).

E4  u_ic = ReLU(pre_ic),  pre_ic = own_ic + ctx_ic + res_ic + b3_c  (exact, fixed point included):
      own_ic = Σ_ch W3[ch,c] m_i,ch q(s_i,ch)     (the particle's own second-layer statements)
      ctx_ic = Σ_ch W3[ch,c] m_i,ch q(g_ch)       (the jet context; m = l3 input bits > 0 for slot i, channel ch)
      res_ic = Σ_ch W3[ch,c] (T − m(q(s)+q(g)))   (the l3 input quantizer's rounding / wrap of the sum)
    variance of own and ctx over real particles, split within jets / between jets; on/off flips of u when g -> its mean;
    causal: g -> mean (all units, then one unit at a time), g -> 0, g -> another jet's g; accuracy and % of calls changed.
    the important g units named with Q observables (best single, greedy 3-term, held-out R²) and with P (variance share).
E5  per S unit with cost >= 0.5%: per-particle response f_c = u_c with g fixed at its mean, tabulated in (pT, r), its
    dependence on azimuth and slot, a hinge fit ReLU(a·pT or a·ln pT + b·r + c) in physical units, value per empty slot;
    naming S_c with Q and with hand-built energy-flow quantities (counts / ΣpT above pT thresholds, in r annuli, ΣpT r^β).
    The causal edits are in e5_causal_jedi.py (they import the helpers below).
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.jedi_core import Jedi, sample

OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'jedi'
SEED = 0
PT_T = [0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 50, 100]          # GeV
R_E = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3]            # r = sqrt(Δη² + Δφ²)
PE = [0, 1, 2, 3, 5, 10, 20, 50, 100, 1e5]                        # table bins
RE = [0, .025, .05, .1, .15, .2, .3, 1.0]


# ------------------------------------------------------------------ helpers (also used by e5_causal_jedi.py)
def bits(layer):
    q = layer.iq.quantizer
    return np.asarray(q.k) + np.asarray(q.i) + np.asarray(q.f)


def masks(net):
    """M3[slot, ch]: l3 input kept; KEEP[slot, c]: u_c of that slot enters the pool S_c"""
    return (bits(net.L['l3']).reshape(64, 64) > 0), (bits(net.L['qs1']).reshape(64, 64) > 0)


def run_pj(net, x, gpj=None, keep=('S', 'logits'), batch=5000):
    """forward pass with g replaced per jet (gpj (J, 64)) or not replaced (None)"""
    out = {k: [] for k in keep}
    one = np.ones(64, np.float32)
    for i in range(0, len(x), batch):
        O = {} if gpj is None else {'g': (one, np.asarray(gpj[i:i + batch], np.float32)[:, None, :])}
        r = net._f(net.inputs(x[i:i + batch], net.n), O)
        for k in keep: out[k].append(np.asarray(r[k], np.float64).reshape(len(x[i:i + batch]), *np.asarray(r[k]).shape[1:]))
    return {k: np.concatenate(v) for k, v in out.items()}


def run_ov(net, x, O, keep=('S', 'logits'), batch=5000):
    """forward pass with per-jet overrides O = {stage: (mask (units,), values (J, units))}; stage 'g' gets (J, 1, 64)"""
    out = {k: [] for k in keep}
    for i in range(0, len(x), batch):
        Ob = {}
        for k, (m, v) in O.items():
            vb = np.asarray(v[i:i + batch], np.float32)
            Ob[k] = (np.asarray(m, np.float32), vb[:, None, :] if k == 'g' else vb)
        r = net._f(net.inputs(x[i:i + batch], net.n), Ob)
        for k in keep: out[k].append(np.asarray(r[k], np.float64).reshape(len(x[i:i + batch]), *np.asarray(r[k]).shape[1:]))
    return {k: np.concatenate(v) for k, v in out.items()}


def make_split(net, live_u):
    """jitted exact split of the l3 pre-activation for units live_u; gov: None or (B, 64) context to use instead of g"""
    import jax, jax.numpy as jnp
    L = net.L
    KB = {k: (jnp.asarray(K, jnp.float32), jnp.asarray(b, jnp.float32)) for k, (K, b) in net.KB.items()}
    M3, _ = masks(net); m3 = jnp.asarray(M3, jnp.float32)
    W3 = KB['l3'][0][:, live_u]; b3 = KB['l3'][1][live_u]

    def dense(k, t):
        return jax.nn.relu(jnp.einsum(L[k].equation, L[k].iq(t, training=False), KB[k][0]) + KB[k][1])

    def f(xin, gov, use):
        a = dense('l1', xin)
        P = jnp.sum(L['qs'].iq(a, training=False), axis=L['qs'].axes, keepdims=L['qs'].keepdims) * L['qs'].scale
        g = dense('dd', P); s = dense('l2', a)
        g = jnp.where(use, gov[:, None, :], g)
        qs_, qd_ = L['add'].iq([s, g], training=False)
        T = L['l3'].iq(qs_ + qd_, training=False)
        own = jnp.einsum('bnc,cC->bnC', m3 * qs_, W3)
        ctx = jnp.einsum('bnc,cC->bnC', m3 * qd_, W3)
        pre = jnp.einsum('bnc,cC->bnC', T, W3) + b3
        return own, ctx, pre - b3 - own - ctx, pre, P.reshape(-1, 64), g.reshape(-1, 64)
    fj = jax.jit(f)

    def run(x, gov=None, batch=4000):
        outs = []
        for i in range(0, len(x), batch):
            xb = net.inputs(x[i:i + batch], net.n)
            gv = np.zeros((len(xb), 64), np.float32) if gov is None else np.broadcast_to(np.asarray(gov, np.float32), (len(xb), 64)) \
                if np.ndim(gov) == 1 else np.asarray(gov[i:i + batch], np.float32)
            outs.append([np.asarray(v, np.float32) for v in fj(xb, gv, gov is not None)])
        return [np.concatenate([o[k] for o in outs]) for k in range(6)]
    return run


def make_masked(net, live_u):
    """float forward that keeps only the zero-bit masks of every input quantizer (no rounding, no wrap);
    returns run(x, gov=None) -> dict(u (J, 64, len(live_u)), g, S, logits); gov replaces g ((64,) or (J, 64))"""
    import jax, jax.numpy as jnp
    L = net.L
    KB = {k: (jnp.asarray(K, jnp.float32), jnp.asarray(b, jnp.float32)) for k, (K, b) in net.KB.items()}

    def qmask(layer):
        qq = layer.iq.quantizers if hasattr(layer.iq, 'quantizers') else [layer.iq]
        return [jnp.asarray(np.asarray(q.quantizer.b) > 0, jnp.float32) for q in qq]
    m = {k: qmask(L[k]) for k in ('l1', 'qs', 'dd', 'l2', 'add', 'l3', 'qs1', 'd4', 'd5', 'd6', 'd7')}
    lu = jnp.asarray(live_u)

    def dn(k, t, act=True):
        z = jnp.einsum(L[k].equation, t * m[k][0], KB[k][0]) + KB[k][1]
        return jax.nn.relu(z) if act else z

    def f(xin, gov, use):
        a = dn('l1', xin)
        P = jnp.sum(a * m['qs'][0], axis=1, keepdims=True) * L['qs'].scale
        g = dn('dd', P); g = jnp.where(use, gov[:, None, :], g)
        s = dn('l2', a)
        u = dn('l3', s * m['add'][0] + g * m['add'][1])
        S = jnp.sum(u * m['qs1'][0], axis=1) * L['qs1'].scale
        z = dn('d6', dn('d5', dn('d4', S)), act=False)
        lo = dn('d7', jax.nn.relu(z), act=False)
        return u[..., lu], g.reshape(-1, 64), S, lo
    fj = jax.jit(f)

    def run(x, gov=None, batch=4000):
        outs = []
        for i in range(0, len(x), batch):
            xb = net.inputs(x[i:i + batch], net.n)
            gv = np.zeros((len(xb), 64), np.float32) if gov is None else np.broadcast_to(np.asarray(gov, np.float32), (len(xb), 64)) \
                if np.ndim(gov) == 1 else np.asarray(gov[i:i + batch], np.float32)
            outs.append([np.asarray(v, np.float32) for v in fj(xb, gv, gov is not None)])
        return {k: np.concatenate([o[j] for o in outs]) for j, k in enumerate(('u', 'g', 'S', 'logits'))}
    return run


def handbuilt(x, slots=None, prefix=''):
    """energy-flow quantities of the real particles (optionally only those in the given slots); dict name -> (J,)"""
    pt, eta, phi = (x[..., k].astype(np.float64) for k in range(3))
    w = pt > 0
    if slots is not None: w = w & slots[None, :]
    r = np.hypot(eta, phi); lp = np.log(np.maximum(pt, 1e-3))
    F = {}
    F['N'] = w.sum(1); F['ΣpT'] = (pt * w).sum(1)
    for t in PT_T:
        a = w & (pt > t); F[f'N(pT>{t})'] = a.sum(1); F[f'ΣpT(pT>{t})'] = (pt * a).sum(1)
    for R in R_E:
        a = w & (r < R); F[f'N(r<{R})'] = a.sum(1); F[f'N(r>{R})'] = (w & ~(r < R)).sum(1)
        F[f'ΣpT(r<{R})'] = (pt * a).sum(1); F[f'ΣpT(r>{R})'] = (pt * (w & ~(r < R))).sum(1)
        F[f'Σ(r-{R})+'] = (np.maximum(r - R, 0) * w).sum(1); F[f'Σ({R}-r)+'] = (np.maximum(R - r, 0) * w).sum(1)
        F[f'ΣpT(r-{R})+'] = (pt * np.maximum(r - R, 0) * w).sum(1)
        F[f'Σln(pT)(r>{R})'] = (lp * (w & ~(r < R))).sum(1)
    ed = [0] + R_E + [9]
    for lo, hi in zip(ed[:-1], ed[1:]):
        a = w & (r >= lo) & (r < hi); F[f'N(r∈[{lo},{hi}))'] = a.sum(1); F[f'ΣpT(r∈[{lo},{hi}))'] = (pt * a).sum(1)
    for b in (0.5, 1, 1.5, 2):
        F[f'ΣpT·r^{b}'] = (pt * r ** b * w).sum(1); F[f'Σr^{b}'] = (r ** b * w).sum(1)
    for t in (1, 2, 5, 10, 20):
        for R in (0.05, 0.1, 0.2):
            F[f'N(pT>{t},r<{R})'] = (w & (pt > t) & (r < R)).sum(1); F[f'N(pT<{t},r>{R})'] = (w & (pt < t) & (r > R)).sum(1)
            F[f'ΣpT(pT>{t},r<{R})'] = (pt * (w & (pt > t) & (r < R))).sum(1)
    if slots is not None: F['N_empty'] = (~(pt > 0) & slots[None, :]).sum(1)
    return {prefix + k: v.astype(np.float64) for k, v in F.items()}


def hinge_eval(p, pt, r):
    """p = dict(form, a, b, c): f = ReLU(a·pT + b·r + c) or ReLU(a·ln pT + b·r + c)"""
    u = np.log(np.maximum(pt, 1e-3)) if p['form'] == 'ln' else pt
    return np.maximum(p['a'] * u + p['b'] * r + p['c'], 0)


def hinge_sum(p, f_pad, keep_slots, x):
    """Σ over the unit's pooled slots: hinge on real particles, f_pad per empty slot; / 16 like the pool"""
    pt = x[..., 0].astype(np.float64); r = np.hypot(x[..., 1], x[..., 2]).astype(np.float64)
    real = pt > 0
    v = np.where(real, hinge_eval(p, pt, r), f_pad) * keep_slots[None, :]
    return v.sum(1) / 16


def r2(pred, t):
    return float(1 - ((pred - t) ** 2).sum() / ((t - t.mean()) ** 2).sum())


def lin_fit(X, t):
    A = np.c_[X, np.ones(len(X))]
    coef = np.linalg.lstsq(A, t, rcond=None)[0]
    return coef, A @ coef


def greedy(F, t, tr, te, k=3):
    """forward selection of k linear terms chosen on the fit half tr; R² reported on the held-out half te"""
    names = [n for n in F if np.std(F[n][tr]) > 1e-9]
    X = np.stack([F[n] for n in names], 1); mu, sd = X[tr].mean(0), X[tr].std(0); Z = (X - mu) / sd
    chosen, steps = [], []
    for _ in range(k):
        best = None
        for j in range(len(names)):
            if j in chosen: continue
            coef, pr = lin_fit(Z[tr][:, chosen + [j]], t[tr])
            sc = r2(pr, t[tr])
            if best is None or sc > best[0]: best = (sc, j)
        chosen.append(best[1])
        coef, _ = lin_fit(Z[tr][:, chosen], t[tr])
        pte = np.c_[Z[te][:, chosen], np.ones(len(te))] @ coef
        raw = coef[:-1] / sd[chosen]; icpt = coef[-1] - (raw * mu[chosen]).sum()
        steps.append(dict(terms=[names[j] for j in chosen], coef=raw.tolist(), intercept=float(icpt),
                          r2_fit=float(best[0]), r2_test=r2(pte, t[te])))
    return steps


def describe_hinge(p):
    d = {}
    if p['form'] == 'lin':
        d['pT_boundary_at_r0'] = float(-p['c'] / p['a']) if p['a'] != 0 else None
        for pt in (1, 10, 100): d[f'r_boundary_at_pT{pt}'] = float(-(p['c'] + p['a'] * pt) / p['b']) if p['b'] != 0 else None
    else:
        d['pT_boundary_at_r0'] = float(np.exp(-p['c'] / p['a'])) if p['a'] != 0 else None
        for pt in (1, 10, 100): d[f'r_boundary_at_pT{pt}'] = float(-(p['c'] + p['a'] * np.log(pt)) / p['b']) if p['b'] != 0 else None
    return d


def fit_hinge(pt, r, f, rng):
    from scipy.optimize import least_squares
    idx = rng.choice(len(f), min(len(f), 80000), replace=False)
    best, per_form = None, {}
    for form in ('lin', 'ln'):
        u = np.log(np.maximum(pt, 1e-3)) if form == 'ln' else pt
        A = np.stack([u, r], 1)[idx]; t = f[idx]; sc = A.std(0) + 1e-12
        on = t > 0
        inits = []
        if on.sum() > 10:
            c0 = np.linalg.lstsq(np.c_[A[on] / sc, np.ones(on.sum())], t[on], rcond=None)[0]; inits.append(c0)
        for s0 in (1, -1):
            for s1 in (1, -1):
                inits.append(np.array([s0 * f.std(), s1 * f.std(), f.mean()]))
        for p0 in inits:
            o = least_squares(lambda p: np.maximum(A / sc @ p[:2] + p[2], 0) - t, p0)
            p = dict(form=form, a=float(o.x[0] / sc[0]), b=float(o.x[1] / sc[1]), c=float(o.x[2]))
            sc_all = r2(hinge_eval(p, pt, r), f)
            if best is None or sc_all > best[1]: best = (p, sc_all)
            if sc_all > per_form.get(form, (None, -9))[1]: per_form[form] = (p, sc_all)
    return best[0], best[1], {k: v[1] for k, v in per_form.items()}


def cellR2(f, *labs):
    key = np.zeros(len(f), np.int64)
    for l in labs: key = key * 1000 + l
    _, inv = np.unique(key, return_inverse=True); cnt = np.bincount(inv); mu = np.bincount(inv, f) / cnt
    return float(1 - ((f - mu[inv]) ** 2).sum() / ((f - f.mean()) ** 2).sum())


def anova(v, w):
    """particle-weighted variance of v (J, n) over the particles w, split into within-jet and between-jet parts"""
    n = w.sum(1); N = n.sum()
    mj = (v * w).sum(1) / np.maximum(n, 1); m = (v * w).sum() / N
    within = float((((v - mj[:, None]) ** 2) * w).sum() / N); between = float((n * (mj - m) ** 2).sum() / N)
    return dict(total=within + between, within=within, between=between, mean=float(m))


def clean_Q(Q, J):
    out = {}
    for k, v in Q.items():
        v = np.asarray(v, np.float64)
        if v.shape != (J,): continue
        bad = ~np.isfinite(v)
        if bad.mean() > 0.01: continue
        if bad.any(): v = np.where(bad, np.median(v[~bad]), v)
        if v.std() > 1e-12: out[k] = v
    return out


# ------------------------------------------------------------------ main
def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True); np.seterr(all='ignore')
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    x, y, Q = sample(per, seed=SEED)
    J = len(x); real = x[..., 0] > 0
    net = Jedi(64)
    M3, KEEP = masks(net)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(J); tr, te = np.sort(perm[:J // 2]), np.sort(perm[J // 2:])
    Qc = clean_Q(Q, J)
    R = net.run(x, keep=('P', 'g', 'S', 'logits'))
    call = R['logits'].argmax(1); acc = float((call == y).mean())
    print(f'{J} jets ({per}/class, seed {SEED}), {real.sum(1).mean():.1f} real particles/jet, accuracy {acc:.4f}; '
          f'{len(Qc)} Q observables usable', flush=True)
    res = dict(N=J, per_class=per, seed=SEED, acc=acc, E4={}, E5={})

    # input wrap check (l1 input quantizer is per slot; WRAP overflow)
    q = net.L['l1'].iq.quantizer
    k_, i_, f_ = (np.asarray(getattr(q, a)).reshape(64, 3) for a in ('k', 'i', 'f'))
    hi = 2.0 ** i_ - 2.0 ** -f_; lo = np.where(k_ > 0, -2.0 ** i_, 0.0)
    xn = net.inputs(x, 64)
    out_rng = ((xn > hi[None]) | (xn < lo[None])) & real[..., None]
    res['input_wrap'] = dict(frac_particles=[float(out_rng[..., k].sum() / real.sum()) for k in range(3)],
                             frac_jets_any=float(out_rng.any((1, 2)).mean()),
                             slot1_pt_below_unsigned_floor=float((real[:, 1] & (x[:, 1, 0] < 14.874)).mean()))
    print('input values outside their slot quantizer range (wrap): particles pT/Δη/Δφ', res['input_wrap']['frac_particles'],
          'jets with any', res['input_wrap']['frac_jets_any'], flush=True)

    # ================================================================ E4
    g = R['g']; gm = g.mean(0)
    live_g = np.flatnonzero(g.std(0) > 1e-9)
    S = R['S']; live_S = np.flatnonzero(S.std(0) > 1e-9)
    split = make_split(net, live_S)
    own, ctx, rs, pre, P2, g2 = split(x)
    assert np.abs(g2 - g).max() < 1e-4
    own_m, ctx_m, rs_m, pre_m, _, _ = split(x, gov=gm)
    E4 = res['E4']
    E4['live_g'] = live_g.tolist(); E4['live_u'] = live_S.tolist()
    E4['l3_input_slots_kept_per_g'] = {int(c): int(M3[:, c].sum()) for c in live_g}
    E4['qs1_slots_kept_per_u'] = {int(c): int(KEEP[:, c].sum()) for c in live_S}

    def edit(name, gnew):
        lo = run_pj(net, x, gnew, keep=('logits',))['logits'] if np.ndim(gnew) == 2 else \
            net.run(x, override={'g': (np.ones(64), gnew)})['logits']
        c2 = lo.argmax(1)
        return dict(edit=name, acc=float((c2 == y).mean()), changed=float((c2 != call).mean()))
    sh = rng.permutation(J)
    E4['causal_all'] = [edit('g -> mean over jets', gm), edit('g -> 0', np.zeros(64)), edit('g -> g of a random other jet', g[sh])]
    for e in E4['causal_all']: print(f"E4 {e['edit']:32s}: accuracy {acc:.4f} -> {e['acc']:.4f}, calls changed {e['changed']*100:.2f}%", flush=True)
    single = []
    for c in live_g:
        m = np.zeros(64); m[c] = 1
        c2 = net.run(x, override={'g': (m, gm)})['logits'].argmax(1)
        single.append(dict(g=int(c), acc=float((c2 == y).mean()), changed=float((c2 != call).mean()), sd=float(g[:, c].std()),
                           on_frac=float((g[:, c] > 0).mean())))
    single.sort(key=lambda d: -d['changed']); E4['causal_single'] = single
    print('E4 g -> mean one unit at a time (unit: % changed, acc):', ', '.join(f"g{d['g']}:{d['changed']*100:.2f}%/{d['acc']:.4f}" for d in single), flush=True)
    # S under mean context
    Sm = net.run(x, override={'g': (np.ones(64), gm)}, keep=('S',))['S']

    # per u unit: variance split, flips
    units = []
    for j, c in enumerate(live_S):
        d = dict(u=int(c))
        for nm, v in (('own', own[..., j]), ('ctx', ctx[..., j]), ('res', rs[..., j]), ('pre', pre[..., j])):
            d[nm] = anova(v.astype(np.float64), real)
        d['b3'] = float(net.KB['l3'][1][c])
        d['cov_own_ctx'] = float(np.cov(own[..., j][real], ctx[..., j][real])[0, 1])
        on, onm = pre[..., j] > 0, pre_m[..., j] > 0
        kr = real & KEEP[None, :, c]; kp = ~real & KEEP[None, :, c]
        d['on_frac_real'] = float(on[real].mean())
        d['flip_real'] = float((on != onm)[real].mean()); d['flip_real_pooled_slots'] = float((on != onm)[kr].mean())
        d['flip_pad_pooled_slots'] = float((on != onm)[kp].mean()) if kp.any() else None
        d['S_corr_with_meanctx'] = float(np.corrcoef(S[:, c], Sm[:, c])[0, 1])
        d['S_var_share_ctx'] = float(np.var(S[:, c] - Sm[:, c]) / np.var(S[:, c]))
        units.append(d)
    E4['units'] = units
    # which second-round objects carry the context's effect: S_c -> its mean-context value (per jet), others unedited
    path = []
    for d in units:
        c = d['u']; m = np.zeros(64); m[c] = 1
        c2 = run_ov(net, x, {'S': (m, Sm)}, keep=('logits',))['logits'].argmax(1)
        path.append(dict(S=c, changed=float((c2 != call).mean()), acc=float((c2 == y).mean())))
    path.sort(key=lambda d: -d['changed'])
    top4 = [d['S'] for d in path[:4]]; m = np.zeros(64); m[top4] = 1
    c2 = run_ov(net, x, {'S': (m, Sm)}, keep=('logits',))['logits'].argmax(1)
    m = np.zeros(64); m[live_S] = 1
    c3 = run_ov(net, x, {'S': (m, Sm)}, keep=('logits',))['logits'].argmax(1)
    E4['context_path'] = dict(single=path, top4=top4, top4_changed=float((c2 != call).mean()), top4_acc=float((c2 == y).mean()),
                              all_changed=float((c3 != call).mean()), all_acc=float((c3 == y).mean()))
    print('E4 context path: S_c -> mean-context S_c alone (% calls changed / acc):', ', '.join(f"S{d['S']}:{d['changed']*100:.2f}%/{d['acc']:.4f}" for d in path if d['changed'] > 0.001),
          f"| top4 {top4} together {E4['context_path']['top4_changed']*100:.2f}%/{E4['context_path']['top4_acc']:.4f} | all live S {E4['context_path']['all_changed']*100:.2f}%/{E4['context_path']['all_acc']:.4f}", flush=True)
    print('E4 per u unit: SD own [total / within jet / between jets] | SD ctx [total / within / between] | SD res | on | flips real (pooled slots) | S: corr with mean-ctx S, var share of ctx', flush=True)
    for d in units:
        print(f"  u{d['u']:<2d} own {np.sqrt(d['own']['total']):6.2f} {np.sqrt(d['own']['within']):6.2f} {np.sqrt(d['own']['between']):6.2f} | "
              f"ctx {np.sqrt(d['ctx']['total']):6.2f} {np.sqrt(d['ctx']['within']):6.2f} {np.sqrt(d['ctx']['between']):6.2f} | res {np.sqrt(d['res']['total']):5.2f} | "
              f"on {d['on_frac_real']:.2f} | flips {d['flip_real']*100:5.2f}% ({d['flip_real_pooled_slots']*100:5.2f}%) | "
              f"S corr {d['S_corr_with_meanctx']:.3f} ctx share {d['S_var_share_ctx']:.3f}", flush=True)

    # name the important g units
    e2 = json.loads((OUT / 'e1_e3.json').read_text())['E2'] if (OUT / 'e1_e3.json').exists() else None
    Kd, bd = net.KB['dd']
    qP = np.asarray(net.L['dd'].iq(R['P'][:, None, :].astype(np.float32), training=False), np.float64).reshape(J, 64)
    gpre = qP @ Kd + bd
    assert np.abs(np.maximum(gpre, 0) - g).max() < 1e-3, np.abs(np.maximum(gpre, 0) - g).max()
    W3 = net.KB['l3'][0]
    names = []
    for d in single:
        c = d['g']
        if d['changed'] < 0.01: continue
        t = g[:, c]
        sQ = greedy(Qc, t, tr, te, k=3)
        terms = qP * Kd[:, c][None, :]
        share = np.array([np.cov(terms[:, k], gpre[:, c])[0, 1] for k in range(64)]) / np.var(gpre[:, c])
        top = np.argsort(-np.abs(share))[:4]
        corrP = np.array([np.corrcoef(R['P'][:, k], t)[0, 1] if R['P'][:, k].std() > 0 else 0 for k in range(64)])
        topc = np.argsort(-np.abs(corrP))[:3]
        tgt = np.array([abs(W3[c, u]) * t.std() * M3[:, c].sum() / 64 for u in live_S]); tt = np.argsort(-tgt)[:3]
        cut = (lambda k: f"{e2[k]['a_pt']:+.3f}·pT {e2[k]['a_eta']:+.1f}·Δη {e2[k]['a_phi']:+.1f}·Δφ {e2[k]['beta']:+.2f}") if e2 else (lambda k: '')
        nm = dict(g=c, cost=d['changed'], Q_single=sQ[0], Q_greedy3=sQ[2],
                  P_share=[dict(P=int(k), share=float(share[k]), w=float(Kd[k, c]), cut=cut(k)) for k in top],
                  P_corr=[dict(P=int(k), r=float(corrP[k])) for k in topc],
                  u_targets=[dict(u=int(live_S[k]), W3=float(W3[c, live_S[k]])) for k in tt],
                  l3_slots_kept=int(M3[:, c].sum()), on_frac=d['on_frac'],
                  class_mean=[float(t[y == k].mean()) for k in range(5)])
        names.append(nm)
        ps = ', '.join('P%d:%+.2f [%s]' % (q_['P'], q_['share'], q_['cut']) for q_ in nm['P_share'])
        pc = ', '.join('P%d:%+.2f' % (q_['P'], q_['r']) for q_ in nm['P_corr'])
        ut = ', '.join('u%d(W %+.2f)' % (q_['u'], q_['W3']) for q_ in nm['u_targets'])
        print(f"E4 g{c} (cost {d['changed']*100:.1f}%, on {d['on_frac']:.2f}, enters l3 at {nm['l3_slots_kept']} slots; class means g/q/W/Z/t "
              f"{' '.join(f'{v:.2f}' for v in nm['class_mean'])}):\n"
              f"    best Q {sQ[0]['terms'][0]} R² {sQ[0]['r2_test']:.3f}; 3 terms {sQ[2]['terms']} R² {sQ[2]['r2_test']:.3f}\n"
              f"    P shares {ps}\n    corr with P {pc}; main u targets {ut}", flush=True)
    E4['g_named'] = names

    # ================================================================ E5
    Sbar = S.mean(0); cost = {}
    for c in live_S:
        m = np.zeros(64); m[c] = 1
        c2 = net.run(x, override={'S': (m, Sbar)})['logits'].argmax(1)
        cost[int(c)] = dict(changed=float((c2 != call).mean()), acc=float((c2 == y).mean()))
    sel = [c for c in sorted(cost, key=lambda k: -cost[k]['changed']) if cost[c]['changed'] >= 0.005]
    res['E5']['cost'] = cost; res['E5']['selected'] = sel
    print('E5 S units changing >= 0.5% of calls when set to their mean:', ', '.join(f"S{c}:{cost[c]['changed']*100:.1f}%" for c in sel), flush=True)
    um = np.maximum(pre_m, 0)                                   # fixed-point response with g at its mean
    MF = make_masked(net, live_S)                               # masked float: zero-bit masks kept, no rounding / wrap
    Mf = MF(x); gmf = Mf['g'].mean(0); Mm = MF(x, gov=gmf)
    cmf = Mf['logits'].argmax(1)
    res['E5']['masked_float'] = dict(acc=float((cmf == y).mean()), changed_vs_fixed=float((cmf != call).mean()),
                                     acc_meanctx=float((Mm['logits'].argmax(1) == y).mean()))
    print(f"masked float network: accuracy {res['E5']['masked_float']['acc']:.4f}, calls changed vs fixed point "
          f"{res['E5']['masked_float']['changed_vs_fixed']*100:.2f}%; with g at its mean {res['E5']['masked_float']['acc_meanctx']:.4f}", flush=True)
    uc = Mm['u']                                                # clean response f_c: masked float, g at its mean
    pt = x[..., 0].astype(np.float64); eta = x[..., 1].astype(np.float64); phi = x[..., 2].astype(np.float64)
    r = np.hypot(eta, phi); th = np.arctan2(phi, eta); slot = np.broadcast_to(np.arange(64), real.shape)
    HB_all = handbuilt(x)
    units5 = []
    ctx_only = [d['u'] for d in units if d['own']['total'] < 1e-12]
    res['E5']['context_only'] = ctx_only
    print('E5 units with no own part (functions of g only):', ctx_only, flush=True)
    for c in sel:
        j = int(np.flatnonzero(live_S == c)[0]); ks = KEEP[:, c]
        if c in ctx_only:
            t = S[:, c]; gQ = greedy(Qc, t, tr, te, k=3)
            Fg = {f'g{k}': g[:, k] for k in live_g}; gG = greedy(Fg, t, tr, te, k=3)
            coef, pr = lin_fit(g[tr][:, live_g], t[tr]); r2g = r2(np.c_[g[te][:, live_g], np.ones(len(te))] @ coef, t[te])
            wts = sorted(((int(k), float(W3[k, c]), float(W3[k, c] * g[:, k].std() * M3[:, k][ks].mean())) for k in live_g), key=lambda z: -abs(z[2]))
            d = dict(S=c, cost=cost[c]['changed'], acc_ablated=cost[c]['acc'], context_only=True, pooled_slots=np.flatnonzero(ks).tolist(),
                     n_pooled_slots=int(ks.sum()), Q_single=gQ[0], Q_greedy3=gQ[2], g_greedy3=gG[2], g_linear_all_R2=r2g, g_weights=wts[:4],
                     class_mean=[float(t[y == k].mean()) for k in range(5)])
            units5.append(d)
            print(f"\nE5 S{c}: cost {d['cost']*100:.1f}%; no own part: S{c} = Σ over {int(ks.sum())} slots of ReLU(W3·g + b3)/16; "
                  f"linear in all live g R² {r2g:.3f}; g 3 {gG[2]['terms']} {gG[2]['r2_test']:.3f}; inputs (g, W3, W3·sd·slot share) {wts[:4]}\n"
                  f"     Q best {gQ[0]['terms'][0]} {gQ[0]['r2_test']:.3f}; Q 3 {gQ[2]['terms']} {gQ[2]['r2_test']:.3f}", flush=True)
            continue
        w = real & ks[None, :]; pad = ~real & ks[None, :]
        f = uc[..., j][w].astype(np.float64); ffx = um[..., j][w].astype(np.float64); fp, rp, ep, hp, sp_ = pt[w], r[w], eta[w], th[w], slot[w]
        pb = np.digitize(fp, PE) - 1; rb = np.digitize(rp, RE) - 1; tb = ((hp + np.pi) / (2 * np.pi) * 8).astype(int) % 8
        sb = np.digitize(sp_, [1, 2, 4, 8, 16, 32])
        tab = [[float(f[(pb == a) & (rb == b)].mean()) if ((pb == a) & (rb == b)).sum() >= 20 else None for b in range(len(RE) - 1)] for a in range(len(PE) - 1)]
        ontab = [[float((f[(pb == a) & (rb == b)] > 0).mean()) if ((pb == a) & (rb == b)).sum() >= 20 else None for b in range(len(RE) - 1)] for a in range(len(PE) - 1)]
        R2s = dict(pT=cellR2(f, pb), r=cellR2(f, rb), pT_r=cellR2(f, pb, rb), pT_r_azimuth=cellR2(f, pb, rb, tb), pT_r_slot=cellR2(f, pb, rb, sb))
        ann = (rp > 0.1) & (rp < 0.3)
        octs = [float(f[ann & (tb == k)].mean()) for k in range(8)]
        hp_, hr2, hforms = fit_hinge(fp, rp, f, np.random.default_rng(SEED))
        hr2_fx = r2(hinge_eval(hp_, fp, rp), ffx); agree = r2(f, ffx)
        fpad = float(uc[..., j][pad].mean()) if pad.any() else 0.0
        fpad_sd = float(uc[..., j][pad].std()) if pad.any() else 0.0
        on = f > 0
        Sg = Sm[:, c]; Sgc = Mm['S'][:, c]
        nact = ((uc[..., j] > 0) & ks[None, :]).sum(1)
        fbar = float(uc[..., j][(uc[..., j] > 0) & ks[None, :]].mean()) if nact.sum() else 0.0
        cnt_part = nact * fbar / 16
        share_count = float(np.cov(cnt_part, Sgc)[0, 1] / np.var(Sgc))
        # pT-linearity among active particles: R² of f on pT (linear) vs on ln pT vs on r
        lin_pt = r2(lin_fit(fp[on][:, None], f[on])[1], f[on]) if on.sum() > 10 else None
        lin_lnpt = r2(lin_fit(np.log(fp[on])[:, None], f[on])[1], f[on]) if on.sum() > 10 else None
        lin_r = r2(lin_fit(rp[on][:, None], f[on])[1], f[on]) if on.sum() > 10 else None
        # naming
        t = S[:, c]
        HB = dict(HB_all)
        if ks.sum() < 64: HB.update(handbuilt(x, ks, prefix='K:'))
        gQ = greedy(Qc, t, tr, te, k=3); gH = greedy(HB, t, tr, te, k=3)
        F_h = hinge_sum(hp_, fpad, ks, x)
        coef, _ = lin_fit(F_h[tr][:, None], t[tr])
        d = dict(S=c, cost=cost[c]['changed'], acc_ablated=cost[c]['acc'], pooled_slots=np.flatnonzero(ks).tolist(),
                 n_pooled_slots=int(ks.sum()), on_frac_real=float(on.mean()), f_pad=fpad, f_pad_sd=fpad_sd,
                 table_mean=tab, table_on=ontab, pT_edges=PE, r_edges=RE, R2_cells=R2s, azimuth_octants_r_0p1_0p3=octs,
                 hinge=hp_, hinge_R2=hr2, hinge_R2_forms=hforms, hinge_R2_vs_fixedpoint_f=hr2_fx, f_clean_vs_fixedpoint_R2=agree,
                 hinge_desc=describe_hinge(hp_), share_count=share_count,
                 active_R2=dict(pT=lin_pt, lnpT=lin_lnpt, r=lin_r),
                 Q_single=gQ[0], Q_greedy3=gQ[2], H_single=gH[0], H_greedy3=gH[2],
                 hinge_sum_R2_raw=r2(F_h[te], t[te]), hinge_sum_R2_rescaled=r2(np.c_[F_h[te], np.ones(len(te))] @ coef, t[te]),
                 hinge_sum_rescale=coef.tolist(),
                 hinge_sum_R2_vs_maskedfloat_S=r2(F_h[te], Mf['S'][te, c]), maskedfloat_S_R2=r2(Mf['S'][te, c], t[te]),
                 meanctx_S_R2=r2(Sg[te], t[te]), meanctx_S_R2_rescaled=float(np.corrcoef(Sg[te], t[te])[0, 1] ** 2),
                 class_mean=[float(t[y == k].mean()) for k in range(5)])
        units5.append(d)
        hd = d['hinge_desc']; hpar = d['hinge']
        print(f"\nE5 S{c}: cost {d['cost']*100:.1f}% (acc {d['acc_ablated']:.4f}); pools {d['n_pooled_slots']} slots"
              f"{'' if ks.sum() == 64 else ' ' + str(_runs(np.flatnonzero(ks)))}; real on {d['on_frac_real']:.2f}; f(empty slot) {fpad:.3f}±{fpad_sd:.3f}\n"
              f"   R² of f by cells: pT {R2s['pT']:.2f}, r {R2s['r']:.2f}, pT×r {R2s['pT_r']:.2f}, +azimuth {R2s['pT_r_azimuth']:.2f}, +slot {R2s['pT_r_slot']:.2f}; "
              f"octant means (r .1–.3) {' '.join(f'{v:.2f}' for v in octs)}\n"
              f"   clean f vs fixed-point f R² {agree:.3f}\n"
              f"   hinge ReLU({hpar['a']:+.4f}·{'ln pT' if hpar['form']=='ln' else 'pT'} {hpar['b']:+.2f}·r {hpar['c']:+.3f}) R² {hr2:.3f} "
              f"(forms {', '.join(f'{k_} {v_:.3f}' for k_, v_ in hforms.items())}; vs fixed-point f {hr2_fx:.3f}); "
              f"pT boundary at r=0: {hd['pT_boundary_at_r0']:.3g} GeV; r boundary at pT 1/10/100 GeV: "
              f"{hd['r_boundary_at_pT1']:.3f} {hd['r_boundary_at_pT10']:.3f} {hd['r_boundary_at_pT100']:.3f}\n"
              f"   active particles: R² of f on pT {lin_pt:.2f}, ln pT {lin_lnpt:.2f}, r {lin_r:.2f}; count term share of S variance {share_count:.2f}\n"
              f"   S naming (held-out R²): hinge sum {d['hinge_sum_R2_raw']:.3f} (rescaled {d['hinge_sum_R2_rescaled']:.3f}; vs masked-float S {d['hinge_sum_R2_vs_maskedfloat_S']:.3f}); "
              f"masked-float S {d['maskedfloat_S_R2']:.3f}; mean-context S {d['meanctx_S_R2']:.3f}\n"
              f"     Q best {gQ[0]['terms'][0]} {gQ[0]['r2_test']:.3f}; Q 3 {gQ[2]['terms']} {gQ[2]['r2_test']:.3f}\n"
              f"     H best {gH[0]['terms'][0]} {gH[0]['r2_test']:.3f}; H 3 {gH[2]['terms']} {gH[2]['r2_test']:.3f}", flush=True)
        print('   table mean f (rows pT', PE[:-1], 'cols r', RE[:-1], ')')
        for a, row in enumerate(tab): print('     ', ' '.join('  -   ' if v is None else f'{v:6.2f}' for v in row))
    res['E5']['units'] = units5
    (OUT / 'e4_e5.json').write_text(json.dumps(res, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    print(f'done {time.time()-t0:.0f}s', flush=True)


def _runs(idx):
    out, s = [], None
    for k in idx:
        if s is None: s = p = k
        elif k == p + 1: p = k
        else: out.append(f'{s}-{p}' if p > s else f'{s}'); s = p = k
    if s is not None: out.append(f'{s}-{p}' if p > s else f'{s}')
    return ','.join(out)


if __name__ == '__main__':
    main()
