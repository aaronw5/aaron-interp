"""E36 (claims A4, A5, B; exploratory method): smallest call-flipping edits at fixed jet pT, mass and axis, and whether the
extracted programs predict them.

Network: official JEDI-linear 3-feature n64 (inputs pT, deta, dphi about the stored axis).
Search model: the float + masks program (interp/e3m_masks.py: every HGQ zero-bit input entry kept as a 0/1 mask, no rounding;
80.68 % accuracy on sample(2000, 0)), written in jax so that it is differentiable.

Edit parametrisation (per real particle i, padding untouched, particle count fixed):
  pT_i -> pT_i exp(u_i)  (so pT >= 0),  deta_i -> deta_i + 0.02 w_i,  dphi_i -> dphi_i + 0.02 w'_i.
  Edit size  eps^2 = sum_i [(dpT_i / pT_i)^2 + (d eta_i^2 + d phi_i^2) / 0.02^2]: eps = 1 is one particle's pT changed by 100 %,
  or one particle moved by 0.02 rad.
Constraints (projection after every Adam step, Gauss-Newton minimum-norm step in (u, w, w') space, float64): the jet
four-momentum sum_i p_i (massless particles, jet axis at (0, 0)) is held at its unedited value, which fixes jet pT, invariant mass
and E-scheme axis; tolerances checked: |dm|/m < 0.1 %, axis shift < 1 mrad.
The edited particles are re-sorted by pT before every forward pass (the network's slot = pT rank convention).
Search: minimise eps^2 + lam * relu(max_{k != t} logit_k - logit_t + kappa) with Adam (200 steps, lr 0.05, kappa 0.05, lam
adapted per jet: x1.1 while not flipped, x0.9 when flipped); the found edit is the smallest-eps projected point at which the float
program calls the target class t.  Targets: call W -> Z, Z -> W, q -> g, g -> q (primary); call t -> its second-highest logit
(secondary).
Evaluation: every found edit on the exact fixed-point network (jedi_core.Jedi.run): transfer = exact call becomes t.
Controls: (1) 100 random edits per jet of the same eps under the same constraints (Gaussian direction in the tangent space of the
constraints, scaled, then projected); (2) untrained network (5 redraws), same search on 400 jets each; (3) on-manifold: HistGBDT
C2ST (interp.e31_onmanifold_jedi.c2st, observables m, pT, N, tau21, tau32, D2, girth, pT^D, centroid term) of edited jets vs real
jets of the target class matched in (log m, log pT), with unedited and random-edited references; kNN support test of E31v.
Programs: E11 nested hinge program (stored e11_fit_*.json, physics recomputed from the edited particles) and the B6 rank-window
program prog30 (stored terms, coefficients refit by least squares on B6's fit split); baseline MLP(m, pT, tau21, D2, N) trained
on the network's calls of a disjoint pool. Agreement = per jet, program flips to t iff the network does.
Samples: main sample(1000, 3601); replicate sample(1000, 3602) minus overlaps; pool sample(2000, 3603) minus both.
Writes $AI_RESULTS/jedi/e36.json, e36_examples.npz.   Run: python -m interp.e36_counterfactual_flip_jedi [--smoke 40]
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
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'
T0 = time.time()
CL = ['g', 'q', 'W', 'Z', 't']; G_, Q_, W_, Z_, T_ = range(5)
PARTNER = {G_: Q_, Q_: G_, W_: Z_, Z_: W_}
SC = 0.02            # angle unit of the edit metric (rad)
ROBUST = 0.5         # secondary 'robust' edit: smallest edit with float-program margin below -0.5 logit
NAMED_S = {0: 'S0 = sum pT of the 14 hardest (via g)', 1: 'S1 = sum_{ranks>=1} (r-0.2)+', 29: 'S29 = sum pT r^2 + sum r^2 + soft count',
           19: 'S19 = soft multiplicity', 30: 'S30 = sum r over the core (r < 0.165)', 31: 'S31 (g-only)', 58: 'S58 (g-only)',
           43: 'S43 (rank 13|14 window)', 50: 'S50 (radial)'}
LIVE_S = [0, 1, 13, 17, 19, 20, 29, 30, 31, 40, 43, 50, 51, 52, 56, 58, 60, 62, 63]
OBJ = {'sum pT, 14 hardest (S0-like)': 'H:ΣpT[top14]', 'sum_{rank>=1} (r-0.2)+ (S1-like)': 'H:Σ(r−0.2)₊[r1-63]',
       'M2 = sum pT r^2 (S29-like)': 'H:ΣpT·r²[all]', 'sum r^2 (S29 part)': 'H:Σr²[all]', 'N(pT < 2 GeV) (S19-like)': 'H:N(pT<2)[all]',
       'sum r, r < 0.165 (S30-like)': 'H:Σr[r<0.165][all]', 'N(pT > 1 GeV)': 'H:N(pT>1)[all]'}


def log(*a): print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def qci(v, B=None):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return [float(np.quantile(v, .025)), float(np.quantile(v, .975))] if len(v) else [None, None]


def boot_mean(v, B=1000, seed=0):
    v = np.asarray(v, float); n = len(v)
    if n == 0: return dict(mean=None, ci=[None, None], n=0)
    rng = np.random.default_rng(seed); bs = [v[rng.integers(0, n, n)].mean() for _ in range(B)]
    return dict(mean=float(v.mean()), ci=qci(bs), n=int(n))


def boot_diff(a, b, B=1000, seed=0):
    """paired: mean(a - b) with 95 % CI and two-sided bootstrap p for H0 mean = 0"""
    d = np.asarray(a, float) - np.asarray(b, float); n = len(d)
    if n == 0: return dict(mean=None, ci=[None, None], p=None, n=0)
    rng = np.random.default_rng(seed); bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    p = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()) + 1 / B))
    return dict(mean=float(d.mean()), ci=qci(bs), p=p, n=int(n))


# --------------------------------------------------------------------------------------------- float + masks program (jax)
def float_program(net):
    import jax, jax.numpy as jnp
    L = net.L; f32 = jnp.float32
    KB = {k: (jnp.asarray(v[0], f32), jnp.asarray(v[1], f32)) for k, v in net.KB.items()}

    def bits(name):
        q = L[name].iq; qs = q.quantizers if hasattr(q, 'quantizers') else [q]; out = []
        for qq in qs:
            qq = qq.quantizer if hasattr(qq, 'quantizer') else qq
            out.append(np.asarray(qq.b) if hasattr(qq, 'b') else None)
        return out
    MASK = {}
    for name in ('l1', 'qs', 'l2', 'dd', 'l3', 'qs1', 'd4', 'd5', 'd6', 'd7'):
        b = bits(name)[0]
        if b is not None: MASK[name] = jnp.asarray(b > 0, f32)
    SCL = {k: float(np.asarray(L[k].scale)) for k in ('qs', 'qs1')}
    sh = jnp.asarray(np.asarray(net.shift, np.float32)); scl = jnp.asarray(np.asarray(net.scale, np.float32))

    def q(name, t): return t * MASK[name] if name in MASK else t

    def dense(k, t, act=True):
        z = jnp.einsum(L[k].equation, q(k, t), KB[k][0]) + KB[k][1]
        return jax.nn.relu(z) if act and L[k].activation is not None else z

    def pool(k, a): return jnp.sum(q(k, a), axis=L[k].axes, keepdims=L[k].keepdims) * SCL[k]

    def f(x):
        xn = (x - sh) / scl
        a = dense('l1', xn); g = dense('dd', pool('qs', a)); s = dense('l2', a)
        u = dense('l3', s + g); h = dense('d5', dense('d4', pool('qs1', u)))
        return dense('d7', jax.nn.relu(dense('d6', h, act=False))).reshape(x.shape[0], -1)
    return f


# --------------------------------------------------------------------------------------------- edits, constraints
def apply_edit(x0, v):
    """x0 (J, n, 3) float64 unedited (sorted), v (J, n, 3) edit params -> edited particles re-sorted by pT, and the order"""
    real = x0[..., 0] > 0
    pt = np.where(real, x0[..., 0] * np.exp(v[..., 0]), 0.0); de = np.where(real, x0[..., 1] + SC * v[..., 1], x0[..., 1])
    dp = np.where(real, x0[..., 2] + SC * v[..., 2], x0[..., 2])
    x = np.stack([pt, de, dp], -1); order = np.argsort(-pt, axis=1, kind='stable')
    return np.take_along_axis(x, order[..., None], 1), order


def p4(x):
    pt, de, dp = x[..., 0], x[..., 1], x[..., 2]
    return np.stack([pt * np.cos(dp), pt * np.sin(dp), pt * np.sinh(de), pt * np.cosh(de)], -1)


def jet_kin(x):
    P = p4(x).sum(1); pt = np.hypot(P[:, 0], P[:, 1]); m = np.sqrt(np.maximum(P[:, 3] ** 2 - (P[:, :3] ** 2).sum(1), 0))
    eta = np.arcsinh(P[:, 2] / np.maximum(pt, 1e-12)); phi = np.arctan2(P[:, 1], P[:, 0])
    return dict(pt=pt, m=m, eta=eta, phi=phi)


def jac(x0, v):
    """residual-free Jacobian of the jet four-momentum wrt (u, w, w'): (J, 4, n*3), masked to real particles"""
    real = (x0[..., 0] > 0).astype(float)
    pt = x0[..., 0] * np.exp(v[..., 0]) * real; de = x0[..., 1] + SC * v[..., 1]; dp = x0[..., 2] + SC * v[..., 2]
    du = np.stack([pt * np.cos(dp), pt * np.sin(dp), pt * np.sinh(de), pt * np.cosh(de)], -1)
    dw = SC * np.stack([0 * pt, 0 * pt, pt * np.cosh(de), pt * np.sinh(de)], -1)
    dwp = SC * np.stack([-pt * np.sin(dp), pt * np.cos(dp), 0 * pt, 0 * pt], -1)
    Jm = np.stack([du, dw, dwp], 2)                       # (J, n, 3, 4)
    return np.transpose(Jm, (0, 3, 1, 2)).reshape(len(x0), 4, -1)


def fourmom(x0, v):
    real = (x0[..., 0] > 0).astype(float)
    pt = x0[..., 0] * np.exp(v[..., 0]) * real; de = x0[..., 1] + SC * v[..., 1]; dp = x0[..., 2] + SC * v[..., 2]
    return np.stack([pt * np.cos(dp), pt * np.sin(dp), pt * np.sinh(de), pt * np.cosh(de)], -1).sum(1)


def project(x0, v, P0, iters=4):
    """Gauss-Newton minimum-norm steps in v space back to sum p = P0 (float64)"""
    real = (x0[..., 0] > 0)[..., None]; v = np.where(real, v, 0.0); J = len(x0); s = np.maximum(np.hypot(P0[:, 0], P0[:, 1]), 1e-9)[:, None]
    for _ in range(iters):
        r = (fourmom(x0, v) - P0) / s; A = jac(x0, v) / s[..., None]
        AA = A @ np.transpose(A, (0, 2, 1)) + 1e-14 * np.eye(4)
        d = -np.einsum('jkn,jk->jn', A, np.linalg.solve(AA, r[..., None])[..., 0])
        v = v + d.reshape(v.shape)
    return np.where(real, v, 0.0)


def tangent(x0, v, d):
    real = (x0[..., 0] > 0)[..., None]; d = np.where(real, d, 0.0); A = jac(x0, v)
    AA = A @ np.transpose(A, (0, 2, 1)) + 1e-14 * np.eye(4); flat = d.reshape(len(d), -1)
    flat = flat - np.einsum('jkn,jk->jn', A, np.linalg.solve(AA, np.einsum('jkn,jn->jk', A, flat)[..., None])[..., 0])
    return np.where(real, flat.reshape(d.shape), 0.0)


def eps2_parts(x0, v):
    """per-particle squared-norm components (J, n): pT, radial, azimuthal (about the particle's unedited position)"""
    real = x0[..., 0] > 0; r0 = np.hypot(x0[..., 1], x0[..., 2]); safe = r0 > 1e-9
    ue = np.where(safe, x0[..., 1] / np.maximum(r0, 1e-12), 1.0); up = np.where(safe, x0[..., 2] / np.maximum(r0, 1e-12), 0.0)
    rad = v[..., 1] * ue + v[..., 2] * up; tan = -v[..., 1] * up + v[..., 2] * ue
    return dict(pt=np.where(real, np.expm1(v[..., 0]) ** 2, 0), rad=np.where(real, rad ** 2, 0), az=np.where(real, tan ** 2, 0))


def eps_of(x0, v):
    P = eps2_parts(x0, v); return np.sqrt(P['pt'].sum(1) + P['rad'].sum(1) + P['az'].sum(1))


def check_constraints(x0, xe):
    a, b = jet_kin(x0), jet_kin(xe)
    dm = np.abs(b['m'] - a['m']) / np.maximum(a['m'], 1e-9); dax = np.hypot(b['eta'] - a['eta'], np.angle(np.exp(1j * (b['phi'] - a['phi']))))
    dpt = np.abs(b['pt'] - a['pt']) / np.maximum(a['pt'], 1e-9)
    return dm, dax, dpt


# --------------------------------------------------------------------------------------------- the search
def make_step(f, kappa):
    import jax, jax.numpy as jnp

    def loss(v, x0, real, oh, lam):
        pt = jnp.where(real, x0[..., 0] * jnp.exp(v[..., 0]), 0.0)
        de = jnp.where(real, x0[..., 1] + SC * v[..., 1], x0[..., 1]); dp = jnp.where(real, x0[..., 2] + SC * v[..., 2], x0[..., 2])
        x = jnp.stack([pt, de, dp], -1); order = jnp.argsort(-jax.lax.stop_gradient(pt), axis=1)
        x = jnp.take_along_axis(x, order[..., None], 1); lo = f(x)
        lt = (lo * oh).sum(1); lother = (lo - 1e9 * oh).max(1); marg = lother - lt
        nrm = ((jnp.expm1(v[..., 0]) ** 2 + v[..., 1] ** 2 + v[..., 2] ** 2) * real).sum(1)
        return (nrm + lam * jax.nn.relu(marg + kappa)).sum(), (marg, nrm, lo.argmax(1))
    g = jax.jit(jax.value_and_grad(loss, has_aux=True))
    return g


def search(step, x0, tgt, steps=200, lr=0.02, chunk=500, tag=''):
    """-> best v (J, n, 3), found (J,), first-flip step (J,), steps to the found point"""
    import jax.numpy as jnp
    J = len(x0); bestv = np.zeros_like(x0); bestn = np.full(J, np.inf); first = np.full(J, -1); robv = np.zeros_like(x0); robn = np.full(J, np.inf)
    for i0 in range(0, J, chunk):
        sl = slice(i0, min(J, i0 + chunk)); n = sl.stop - sl.start; pad = chunk - n
        xc = np.concatenate([x0[sl], np.repeat(x0[sl][:1], pad, 0)]) if pad else x0[sl]
        tc = np.concatenate([tgt[sl], np.repeat(tgt[sl][:1], pad)]) if pad else tgt[sl]
        real = xc[..., 0] > 0; P0 = p4(xc).sum(1); oh = np.eye(5)[tc]
        v = np.zeros_like(xc); m = np.zeros_like(xc); s2 = np.zeros_like(xc); lam = np.full(len(xc), 0.3)
        bv = np.zeros_like(xc); bn = np.full(len(xc), np.inf); fs = np.full(len(xc), -1); rv = np.zeros_like(xc); rn = np.full(len(xc), np.inf)
        xj, rj, oj = jnp.asarray(xc, jnp.float32), jnp.asarray(real), jnp.asarray(oh, jnp.float32)
        for it in range(steps + 1):
            (_, (marg, nrm, call)), gr = step(jnp.asarray(v, jnp.float32), xj, rj, oj, jnp.asarray(lam, jnp.float32))
            call = np.asarray(call); nrm = eps_of(xc, v) ** 2; fl = call == tc
            upd = fl & (nrm < bn); bv[upd] = v[upd]; bn[upd] = nrm[upd]; fs[(fs < 0) & fl] = it
            ur = fl & (np.asarray(marg) < -ROBUST) & (nrm < rn); rv[ur] = v[ur]; rn[ur] = nrm[ur]
            lam = np.where(fl, lam * 0.9, lam * 1.1)
            if it == steps: break
            gr = np.asarray(gr, np.float64) * real[..., None]
            m = 0.9 * m + 0.1 * gr; s2 = 0.999 * s2 + 0.001 * gr ** 2
            mh = m / (1 - 0.9 ** (it + 1)); sh = s2 / (1 - 0.999 ** (it + 1))
            v = project(xc, v - lr * mh / (np.sqrt(sh) + 1e-8), P0)
            if it % 50 == 0:
                log(f'{tag} jets {sl.start}-{sl.stop}: step {it}: flipped now {fl[:n].mean():.3f}, found so far {np.isfinite(bn[:n]).mean():.3f}, '
                    f'median eps (found) {np.median(np.sqrt(bn[:n][np.isfinite(bn[:n])])) if np.isfinite(bn[:n]).any() else float("nan"):.3f}')
        bestv[sl] = bv[:n]; bestn[sl] = bn[:n]; first[sl] = fs[:n]; robv[sl] = rv[:n]; robn[sl] = rn[:n]
    search.robust = (robv, np.isfinite(robn))
    return bestv, np.isfinite(bestn), first


def random_edits(x0, v_found, R, rng):
    """R random edits per jet at the found edit's v-space norm, tangent to the constraints, then projected; -> list of v"""
    P0 = p4(x0).sum(1); nv = np.sqrt((v_found ** 2).sum((1, 2))); out = []
    for _ in range(R):
        d = tangent(x0, np.zeros_like(x0), rng.normal(size=x0.shape))
        d = d * (nv / np.maximum(np.sqrt((d ** 2).sum((1, 2))), 1e-12))[:, None, None]
        out.append(project(x0, d, P0, iters=30))
    return out


# --------------------------------------------------------------------------------------------- programs
class Programs:
    def __init__(self, net):
        from interp import e11_mars_all as E
        from interp.e11_verify import Last, chain
        from interp.b6_pareto import features, build
        from jetdistill import mars
        self.E, self.chain, self.last, self.mars = E, chain, Last(net), mars
        rd = lambda n: json.loads((OUT / f'e11_{n}.json').read_text())
        self.fb = {'P': rd('fit_a_P'), **{s: rd(f'fit_b_{s}') for s in E.STG[1:]}}
        d = json.loads((OUT / 'b6_pred_jedi.json').read_text()); self.terms = [t[:30] for t in d['prog_terms']]; self.keys = d['observables']
        self.features = features
        B = build(); Xf, keys = features(B['x']['fit']); assert keys == self.keys
        Lf = net.run(B['x']['fit'])['logits']; Q = {k: Xf[:, i] for i, k in enumerate(keys)}; NF = len(Lf)
        self.co = [mars._lstsq(self.terms[k], Q, Lf[:, k], NF) for k in range(5)]
        log(f'B6 prog30 coefficients refit on {NF} fit jets (same call on them {(self.b6(B["x"]["fit"]) == Lf.argmax(1)).mean():.4f})')

    def e11(self, x):
        from jetdistill.observables import compute
        q = compute(np.asarray(x, np.float64), 64); ph = {k: np.asarray(v, np.float32).astype(np.float64) for k, v in q.items()}
        ph.update(self.E.handbuilt(np.asarray(x, np.float32), False)); V = self.chain(self.fb, ph)
        return self.last.jax_logits(V['z']).argmax(1)

    def b6(self, x):
        X, _ = self.features(np.asarray(x, np.float32)); Q = {k: X[:, i].astype(np.float64) for i, k in enumerate(self.keys)}; J = len(X)
        return np.stack([np.stack([np.ones(J)] + [self.mars.term_value(t, Q) for t in self.terms[k]], 1) @ self.co[k] for k in range(5)], 1).argmax(1)


def mlp_baseline(xp, cp, seed=3604):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from interp.e31_onmanifold_jedi import jet_feats
    def feats(x):
        F = jet_feats(x); return np.stack([np.log(np.maximum(F['m'], 1e-3)), np.log(F['jpt']), F['tau21'], np.log(np.maximum(F['D2'], 1e-6)), F['N']], 1)
    X = feats(xp); sc = StandardScaler().fit(X)
    m = MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=400, early_stopping=True, random_state=seed).fit(sc.transform(X), cp)
    return lambda x: m.predict(sc.transform(feats(x))), float(m.best_validation_score_)


# --------------------------------------------------------------------------------------------- one sample
def run_sample(name, x, y, net, f, step, a, rng, R, P=None, mlp=None, pool=None):
    from interp.e28w_split_route_jedi import mcnemar
    from interp.e31_onmanifold_jedi import jet_feats
    import jax.numpy as jnp
    out = dict(N=int(len(y)))
    x0 = np.asarray(x, np.float64); J = len(x0)
    ex0 = net.run(x, keep=('logits', 'S')); c_ex = ex0['logits'].argmax(1)
    lo_f = np.concatenate([np.asarray(f(jnp.asarray(x[i:i + 1000]))) for i in range(0, J, 1000)]); c_f = lo_f.argmax(1)
    out['acc_exact'] = float((c_ex == y).mean()); out['acc_float_masks'] = float((c_f == y).mean()); out['float_exact_same_call'] = float((c_f == c_ex).mean())
    log(f'{name}: N {J}; exact acc {out["acc_exact"]:.4f}; float+masks acc {out["acc_float_masks"]:.4f}; same call {out["float_exact_same_call"]:.4f}')
    srt = np.sort(lo_f, 1); tgt = np.array([PARTNER.get(int(c), -1) for c in c_f])
    sec = np.argsort(-lo_f, 1)[:, 1]; tgt = np.where(tgt < 0, sec, tgt)
    sel = c_f == c_ex                                      # jets where float and exact agree before the edit
    out['n_selected'] = int(sel.sum())
    v, found, first = search(step, x0, tgt, a.steps, a.lr, tag=name)
    out['found_float'] = float(found[sel].mean())
    xe, _ = apply_edit(x0, v); dm, dax, dpt = check_constraints(x0, xe)
    ok = sel & found & (dm < 1e-3) & (dax < 1e-3)
    out['constraint'] = dict(dm_rel_q99=float(np.quantile(dm[found], .99)) if found.any() else None, axis_mrad_q99=float(np.quantile(dax[found], .99) * 1e3) if found.any() else None,
                             dpt_rel_q99=float(np.quantile(dpt[found], .99)) if found.any() else None, n_found=int(found.sum()), n_ok=int(ok.sum()))
    xe32 = xe.astype(np.float32); exE = net.run(xe32, keep=('logits', 'S')); c_e = exE['logits'].argmax(1)
    c_fe = np.concatenate([np.asarray(f(jnp.asarray(xe32[i:i + 1000]))).argmax(1) for i in range(0, J, 1000)])
    eps = eps_of(x0, v)
    grp = {'W->Z': c_f == W_, 'Z->W': c_f == Z_, 'q->g': c_f == Q_, 'g->q': c_f == G_, 't->2nd': c_f == T_}
    prim = ok & (c_f != T_)
    out['float_recheck_flip'] = float((c_fe[ok] == tgt[ok]).mean())
    tr = (c_e == tgt).astype(float)
    out['transfer'] = {'primary': boot_mean(tr[prim]), **{g: boot_mean(tr[ok & m]) for g, m in grp.items()}}
    out['eps'] = {g: dict(median=float(np.median(eps[ok & m])) if (ok & m).any() else None, q25=float(np.quantile(eps[ok & m], .25)) if (ok & m).any() else None,
                          q75=float(np.quantile(eps[ok & m], .75)) if (ok & m).any() else None, found_float=float(found[sel & m].mean()) if (sel & m).any() else None)
                  for g, m in {'primary': c_f != T_, **grp}.items()}
    rv_, rf_ = search.robust; xr_ = apply_edit(x0, rv_)[0]; c_r = net.run(xr_.astype(np.float32))['logits'].argmax(1); rok = sel & rf_ & (c_f != T_)
    er_ = eps_of(x0, rv_)
    out['robust_edit'] = dict(margin=ROBUST, found=float(rf_[sel & (c_f != T_)].mean()), transfer=boot_mean((c_r == tgt)[rok].astype(float)),
                              eps_median=float(np.median(er_[rok])) if rok.any() else None, eps_ratio_to_minimal_median=float(np.median(er_[rok & ok] / eps[rok & ok])) if (rok & ok).any() else None,
                              by_group={g: boot_mean((c_r == tgt)[rok & m].astype(float)) for g, m in grp.items() if (rok & m).sum() >= 20})
    log(f'{name}: robust edits (float margin < -{ROBUST}): found {out["robust_edit"]["found"]:.3f}, transfer {out["robust_edit"]["transfer"]["mean"]}, eps x{out["robust_edit"]["eps_ratio_to_minimal_median"]}')
    out['first_flip_step_median'] = float(np.median(first[found])) if found.any() else None
    log(f'{name}: found {found[sel].mean():.3f}; ok (constraints) {ok.sum()}; transfer primary {out["transfer"]["primary"]["mean"]:.3f} {out["transfer"]["primary"]["ci"]}; '
        f'median eps {out["eps"]["primary"]["median"]}; constraint q99 dm {out["constraint"]["dm_rel_q99"]} axis {out["constraint"]["axis_mrad_q99"]} mrad')
    # ---- random same-norm edits (exact network)
    idx = np.flatnonzero(ok); rflip = np.zeros((len(idx), R)); rany = np.zeros((len(idx), R)); rflip_f = np.zeros((len(idx), R))
    RV0 = None; rS = []; rcomp = np.zeros((len(idx), R), bool); RVc = np.zeros_like(x0[idx]); have = np.zeros(len(idx), bool); rflip_c = np.zeros(len(idx), bool)
    for i0 in range(0, len(idx), 500):
        ii = idx[i0:i0 + 500]; Vs = random_edits(x0[ii], v[ii], R, rng)
        for r, vr in enumerate(Vs):
            xr, _ = apply_edit(x0[ii], vr); xr32 = xr.astype(np.float32); dm_, dax_, _ = check_constraints(x0[ii], xr); cm = (dm_ < 1e-3) & (dax_ < 1e-3)
            n_ = len(ii); new = cm & ~have[i0:i0 + n_]; rcomp[i0:i0 + n_, r] = cm; res = net.run(xr32, keep=('logits', 'S') if r < min(10, R) else ('logits',))
            cr = res['logits'].argmax(1); rflip[i0:i0 + len(ii), r] = cr == tgt[ii]; rany[i0:i0 + len(ii), r] = cr != c_ex[ii]
            blk = RVc[i0:i0 + n_]; blk[new] = vr[new]; fb = rflip_c[i0:i0 + n_]; fb[new] = (cr == tgt[ii])[new]; have[i0:i0 + n_] |= cm
            if r == 0:
                rflip_f[i0:i0 + len(ii), 0] = np.concatenate([np.asarray(f(jnp.asarray(xr32[k:k + 500]))).argmax(1) for k in range(0, len(ii), 500)]) == tgt[ii]
                RV0 = vr if RV0 is None else np.concatenate([RV0, vr])
            if r < min(10, R): rS.append((i0, r, res['S']))
        log(f'{name}: random edits {i0 + len(ii)}/{len(idx)} jets x {R}')
    nS = min(10, R); Sr = np.zeros((len(idx), nS, 64))
    for i0, r, S in rS: Sr[i0:i0 + len(S), r] = np.where(rcomp[i0:i0 + len(S), r][:, None], S, np.nan)
    out['random_constraint_compliance'] = dict(draws=float(rcomp.mean()), jets_with_any=float(have.mean()), primary_jets_with_any=float(have[prim[idx]].mean()),
                                               rule='|dm|/m < 1e-3 and axis shift < 1 mrad after 30 Gauss-Newton steps; non-compliant draws excluded')
    rflip_f[:, 0] = np.where(rcomp[:, 0], rflip_f[:, 0], np.nan); rflip = np.where(rcomp, rflip, np.nan); rany = np.where(rcomp, rany, np.nan)
    RV0 = RVc
    log(f'{name}: random-edit constraint compliance: draws {rcomp.mean():.3f}, jets with >= 1 compliant draw {have.mean():.3f}')
    rv_eps = eps_of(x0[idx], RV0); out['random_eps_over_found_median'] = float(np.median((rv_eps / eps[idx])[have]))
    pm = prim[idx] & have           # primary jets with >= 1 constraint-compliant random draw
    out['random'] = dict(R=R, flip_to_target=boot_mean(np.nanmean(rflip[pm], 1)), any_call_change=boot_mean(np.nanmean(rany[pm], 1)),
                         float_flip_draw0=boot_mean(rflip_f[pm & rcomp[:, 0], 0]),
                         by_group={g: boot_mean(np.nanmean(rflip[m[idx] & have], 1)) for g, m in grp.items()})
    out['found_vs_random'] = dict(diff=boot_diff(tr[idx][pm], np.nanmean(rflip[pm], 1)),
                                  mcnemar_vs_draw0=mcnemar(tr[idx][pm].astype(bool), rflip_c[pm].astype(bool)))
    log(f'{name}: random same-eps flip to target {out["random"]["flip_to_target"]["mean"]:.4f} {out["random"]["flip_to_target"]["ci"]}; any change '
        f'{out["random"]["any_call_change"]["mean"]:.4f}; McNemar {out["found_vs_random"]["mcnemar_vs_draw0"]}')
    # ---- where the norm goes (found vs random draw 0), primary jets
    real = x0[..., 0] > 0; rank = np.broadcast_to(np.arange(64), real.shape); r0 = np.hypot(x0[..., 1], x0[..., 2]); pt0 = x0[..., 0]
    def shares(xx, vv):
        Pp = eps2_parts(xx, vv); tot = sum(Pp.values()).sum(1) + 1e-30; rk = rank[:len(xx)]; rr = np.hypot(xx[..., 1], xx[..., 2]); pp = xx[..., 0]
        allc = sum(Pp.values()); top = np.sort(allc, 1)[:, ::-1]
        return {'coord: pT': Pp['pt'].sum(1) / tot, 'coord: radial': Pp['rad'].sum(1) / tot, 'coord: azimuthal': Pp['az'].sum(1) / tot,
                'ranks 0-13': (allc * (rk < 14)).sum(1) / tot, 'ranks 14-63': (allc * (rk >= 14)).sum(1) / tot,
                'r < 0.1': (allc * (rr < 0.1)).sum(1) / tot, '0.1 <= r < 0.2': (allc * ((rr >= 0.1) & (rr < 0.2))).sum(1) / tot, 'r >= 0.2': (allc * (rr >= 0.2)).sum(1) / tot,
                'radial, ranks 0-13': (Pp['rad'] * (rk < 14)).sum(1) / tot, 'radial, ranks 14-63': (Pp['rad'] * (rk >= 14)).sum(1) / tot,
                'pT < 11 GeV particles': (allc * (pp < 11) * (pp > 0)).sum(1) / tot, 'top-3 particles (largest share)': top[:, :3].sum(1) / tot}
    sf = shares(x0[idx][pm], v[idx][pm]); sr = shares(x0[idx][pm], RV0[pm])
    pc = {'ranks 0-13': ((rank < 14) & real)[idx][pm].sum(1) / real[idx][pm].sum(1)}
    out['norm_shares'] = {k: dict(found=boot_mean(sf[k]), random=boot_mean(sr[k]), diff=boot_diff(sf[k], sr[k])) for k in sf}
    out['norm_shares']['_particle_share_ranks_0_13'] = boot_mean(pc['ranks 0-13'])
    out['norm_shares_by_group'] = {}
    for g, m in grp.items():
        mm = m[idx] & have
        if mm.sum() < 20: continue
        a_, b_ = shares(x0[idx][mm], v[idx][mm]), shares(x0[idx][mm], RV0[mm])
        out['norm_shares_by_group'][g] = {k: dict(found=float(a_[k].mean()), random=float(b_[k].mean())) for k in a_}
    log(f'{name}: shares found/random: ' + ', '.join(f'{k} {out["norm_shares"][k]["found"]["mean"]:.3f}/{out["norm_shares"][k]["random"]["mean"]:.3f}'
                                                    for k in sf))
    # direction of the radial and pT moves on the found edits (signed, pT-weighted), by group
    out['signed'] = {}
    for g, m in grp.items():
        mm = (ok & m)
        if mm.sum() < 20: continue
        vv, xx = v[mm], x0[mm]; rr = np.hypot(xx[..., 1], xx[..., 2]); rl = xx[..., 0] > 0
        dr = (vv[..., 1] * xx[..., 1] + vv[..., 2] * xx[..., 2]) / np.maximum(rr, 1e-12) * SC
        out['signed'][g] = dict(mean_dr_mrad_ranks0_13=float((dr * rl)[:, :14].sum(1).mean() / 14 * 1e3), mean_dr_mrad_ranks14_63=float(((dr * rl)[:, 14:].sum(1) / np.maximum(rl[:, 14:].sum(1), 1)).mean() * 1e3),
                                mean_dlogpt_ranks0_13=float(np.mean(vv[:, :14, 0])), frac_jets_net_outward_hard=float(((dr * rl)[:, :14].sum(1) > 0).mean()))
    # ---- named S units (exact network): |dS| / sd, found vs random (mean over 10 draws)
    sdS = ex0['S'].std(0) + 1e-9; dSf = (exE['S'][idx] - ex0['S'][idx]) / sdS; dSr = (Sr - ex0['S'][idx][:, None]) / sdS
    from interp.e28w_split_route_jedi import holm
    Srows = {}; ps = []
    for u in LIVE_S:
        a_, b_ = np.abs(dSf[pm, u]), np.nanmean(np.abs(dSr[pm, :, u]), 1); d_ = boot_diff(a_, b_, seed=u)
        Srows[f'S{u}'] = dict(name=NAMED_S.get(u, ''), found_abs=float(a_.mean()), random_abs=float(b_.mean()), ratio=float(a_.mean() / max(b_.mean(), 1e-12)),
                              diff=d_, signed_by_group={g: float(dSf[m[idx], u].mean()) for g, m in grp.items() if m[idx].sum() >= 20})
        ps.append(d_['p'])
    for k, p in zip(Srows, holm(ps)): Srows[k]['p_holm'] = float(p)
    out['S_units'] = Srows
    log(f'{name}: |dS|/sd found vs random: ' + ', '.join(f'{k} {v_["found_abs"]:.3f}/{v_["random_abs"]:.3f}' for k, v_ in Srows.items() if int(k[1:]) in NAMED_S))
    # ---- observables and named physics objects
    from interp.e11_mars_all import handbuilt
    F0, FE, FR = jet_feats(x0[idx]), jet_feats(xe[idx]), jet_feats(apply_edit(x0[idx], RV0)[0])
    H0, HE, HR = handbuilt(x0[idx].astype(np.float32)), handbuilt(xe[idx].astype(np.float32)), handbuilt(apply_edit(x0[idx], RV0)[0].astype(np.float32))
    FA = jet_feats(x0)
    obs = {}
    for k in ('m', 'jpt', 'N', 'tau21', 'tau32', 'D2', 'girth', 'ptD'):
        sd = FA[k].std() + 1e-12
        obs[k] = dict(found=boot_mean((FE[k] - F0[k])[pm] / sd), random=boot_mean((FR[k] - F0[k])[pm] / sd), sd=float(sd),
                      by_group={g: float(((FE[k] - F0[k]) / sd)[m[idx]].mean()) for g, m in grp.items() if m[idx].sum() >= 20},
                      class_gap_sd={g: float((FA[k][c_ex == tg].mean() - FA[k][c_ex == sg].mean()) / sd) for g, sg, tg in
                                    (('W->Z', W_, Z_), ('Z->W', Z_, W_), ('q->g', Q_, G_), ('g->q', G_, Q_))})
    HA = handbuilt(np.asarray(x, np.float32))
    for lab, key in OBJ.items():
        sd = HA[key].std() + 1e-12
        obs[lab] = dict(found=boot_mean((HE[key] - H0[key])[pm] / sd), random=boot_mean((HR[key] - H0[key])[pm] / sd), sd=float(sd),
                        by_group={g: float(((HE[key] - H0[key]) / sd)[m[idx]].mean()) for g, m in grp.items() if m[idx].sum() >= 20})
    out['observables_delta_sd'] = obs
    log(f'{name}: d(obs)/sd found: ' + ', '.join(f'{k} {v_["found"]["mean"]:.3f}' for k, v_ in obs.items()))
    # ---- programs
    if P is not None:
        res = {}
        xr0 = apply_edit(x0[idx], RV0)[0].astype(np.float32)
        for pname, fn in (('E11 nested program', P.e11), ('B6 rank-window prog30', P.b6), ('MLP(m, pT, tau21, D2, N)', mlp)):
            cu = fn(x[idx]); ce = fn(xe32[idx]); cr = fn(xr0)
            base_ok = cu == c_ex[idx]
            nf = (c_e[idx] == tgt[idx]); pf = (ce == tgt[idx]) & base_ok
            agree = (nf == pf).astype(float); nr = rflip_c.astype(bool); pr_ = (cr == tgt[idx]) & base_ok; agree_r = (nr == pr_).astype(float)
            res[pname] = dict(same_call_unedited=float(base_ok.mean()), agreement_found=boot_mean(agree[pm]),
                              program_flip_rate_found=boot_mean(pf[pm].astype(float)), recall_on_net_flips=boot_mean(pf[pm & nf].astype(float)),
                              specificity_on_net_nonflips=boot_mean((~pf)[pm & ~nf].astype(float)),
                              mcnemar_net_vs_program=mcnemar(nf[pm], pf[pm]), agreement_random=boot_mean(agree_r[pm]),
                              program_flip_rate_random=boot_mean(pr_[pm].astype(float)),
                              by_group={g: dict(agreement=float(agree[m[idx]].mean()), prog_flip=float(pf[m[idx]].mean()), net_flip=float(nf[m[idx]].mean()))
                                        for g, m in grp.items() if m[idx].sum() >= 20}, _agree_vec=agree[pm])
            log(f'{name}: {pname}: same call unedited {base_ok.mean():.3f}; agreement found {res[pname]["agreement_found"]["mean"]:.3f} {res[pname]["agreement_found"]["ci"]}; '
                f'prog flip {pf[pm].mean():.3f} vs net {nf[pm].mean():.3f}; random agreement {agree_r[pm].mean():.3f}')
        kk = list(res); comp = {}
        for i in range(len(kk)):
            for j in range(i + 1, len(kk)):
                comp[f'{kk[i]} - {kk[j]}'] = dict(diff=boot_diff(res[kk[i]]['_agree_vec'], res[kk[j]]['_agree_vec']),
                                                 mcnemar=mcnemar(res[kk[i]]['_agree_vec'].astype(bool), res[kk[j]]['_agree_vec'].astype(bool)))
        for k in kk: res[k].pop('_agree_vec')
        res['_paired'] = comp; out['programs'] = res
    # ---- on-manifold (main sample)
    if pool is not None and pm.sum() >= 50:
        from interp.e31_onmanifold_jedi import c2st, OBS
        from interp.e31_verify_jedi import knn_support
        from sklearn.neighbors import NearestNeighbors
        xp, yp, Fp_all = pool
        def obsmat(F): return np.nan_to_num(np.stack([F[k] for k in OBS], 1), nan=0.0, posinf=1e6, neginf=-1e6)
        Pm = obsmat(Fp_all); key = np.stack([np.log(np.maximum(Fp_all['m'], 1e-3)), np.log(Fp_all['jpt'])], 1); mu, sd = key.mean(0), key.std(0)
        ii = idx[pm]; FEo, F0o, FRo = obsmat(jet_feats(xe[ii])), obsmat(jet_feats(x0[ii])), obsmat(jet_feats(apply_edit(x0[idx], RV0)[0][pm]))
        k0 = (np.stack([np.log(np.maximum(F0o[:, 0], 1e-3)), np.log(F0o[:, 1])], 1) - mu) / sd
        match = np.zeros(len(ii), int)
        for t in range(5):
            mt = tgt[ii] == t; cand = np.flatnonzero(yp == t)
            if mt.any(): match[mt] = cand[NearestNeighbors(n_neighbors=1).fit((key[cand] - mu) / sd).kneighbors(k0[mt])[1][:, 0]]
        Fmt = Pm[match]; okm = np.ones(len(ii), bool)
        cs = {}
        for lab, ref, q_, sd_ in (('edited vs matched real target-class', Fmt, FEo, 3611), ('unedited vs matched real target-class', Fmt, F0o, 3612),
                                  ('random same-eps vs matched real target-class', Fmt, FRo, 3613), ('edited vs unedited (same jets)', F0o, FEo, 3614),
                                  ('random same-eps vs unedited (same jets)', F0o, FRo, 3615)):
            r_, _, _ = c2st(ref, q_, okm, sd_, 5, a.nboot, a.nperm); cs[lab] = r_
            log(f'{name}: C2ST {lab}: AUC {r_["auc"]:.4f} {r_["ci"]} p {r_["p_swap"]:.4f}')
        out['c2st_obs'] = cs
        out['c2st_features'] = OBS
        half = np.random.default_rng(3616).permutation(len(xp)); A_, B_ = half[:len(xp) // 2], half[len(xp) // 2:]
        sup = {}
        for fs, cols in (('obs8', list(range(8))), ('inv', [0, 1, 2, 3, 4, 5, 7])):
            thr, dh, dq = knn_support(Pm[A_][:, cols], [FEo[:, cols], FRo[:, cols], F0o[:, cols]], Pm[B_][:, cols])
            sup[fs] = dict(threshold=thr, held_out_real_outside=float((dh > thr).mean()),
                           **{lab: boot_mean((d > thr).astype(float)) for lab, d in zip(('edited', 'random same-eps', 'unedited'), dq)})
            log(f'{name}: kNN support {fs}: outside edited {sup[fs]["edited"]["mean"]:.3f}, random {sup[fs]["random same-eps"]["mean"]:.3f}, unedited {sup[fs]["unedited"]["mean"]:.3f}')
        out['knn_support'] = sup
    return out, dict(x0=x0, v=v, xe=xe, ok=ok, prim=prim, tgt=tgt, c_f=c_f, c_e=c_e, eps=eps, idx=idx, RV0=RV0, y=y)


# --------------------------------------------------------------------------------------------- main
def main(a):
    from interp.jedi_core import Jedi, sample
    from interp.e11_mars_all import indices
    from interp.x10_das_jedi import clean
    OUT.mkdir(parents=True, exist_ok=True); sfx = '_smoke' if a.smoke else ''; JS = OUT / f'e36{sfx}.json'
    R = dict(node='E36', samples=dict(main='sample(1000, 3601)', replicate='sample(1000, 3602) minus main', pool='sample(2000, 3603) minus both'),
             settings=dict(steps=a.steps, lr=a.lr, kappa=a.kappa, angle_unit_rad=SC, random_per_jet=a.R, untrained_redraws=a.untrained, untrained_jets=a.unt_n))
    save = lambda: JS.write_text(json.dumps(clean(R), indent=1))
    pc = a.smoke or 1000
    x, y, _ = sample(pc, 3601); i1 = indices(pc, 3601)
    x2, y2, _ = sample(pc, 3602); i2 = indices(pc, 3602); k2 = ~np.isin(i2, i1); x2, y2 = x2[k2], y2[k2]
    xp, yp, _ = sample(2 * pc, 3603); ip = indices(2 * pc, 3603); kp = ~np.isin(ip, np.concatenate([i1, i2])); xp, yp = np.asarray(xp[kp]), yp[kp]
    log(f'jets: main {len(y)}, replicate {len(y2)} ({(~k2).sum()} overlaps removed), pool {len(yp)}')
    net = Jedi(64); f = float_program(net)
    import jax
    fj = jax.jit(f); step = make_step(f, a.kappa)
    from interp.e31_onmanifold_jedi import jet_feats
    cp = net.run(xp)['logits'].argmax(1); mlp, mlp_val = mlp_baseline(xp, cp); R['mlp_baseline'] = dict(features='log m, log pT, tau21, log D2, N', hidden=[32, 32],
                                                                                                      trained_on='pool jets, target = exact network call', val_acc=mlp_val)
    log(f'MLP baseline trained on {len(yp)} pool jets (validation same call {mlp_val:.3f})')
    P = Programs(net); Fp = jet_feats(xp)
    rng = np.random.default_rng(3605)
    R['main'], keep = run_sample('main', x, y, net, fj, step, a, rng, a.R, P=P, mlp=mlp, pool=(xp, yp, Fp)); save()
    R['replicate'], _ = run_sample('replicate', x2, y2, net, fj, step, a, rng, max(10, a.R // 5), P=P, mlp=mlp); save()
    # ---- untrained control
    U = []
    rs = np.random.default_rng(3606); sub = np.sort(rs.choice(len(y), min(a.unt_n, len(y)), replace=False)); x0s = np.asarray(x[sub], np.float64)
    tr_eps = keep['eps'][sub][keep['ok'][sub]]
    for s in range(a.untrained):
        nu = Jedi(64, untrained=True, seed=s); fu = jax.jit(float_program(nu)); su = make_step(float_program(nu), a.kappa)
        lo = np.asarray(fu(jax.numpy.asarray(x[sub]))); cu = lo.argmax(1); tu = np.array([PARTNER.get(int(c), -1) for c in cu]); tu = np.where(tu < 0, np.argsort(-lo, 1)[:, 1], tu)
        vu, fdu, _ = search(su, x0s, tu, a.steps, a.lr, tag=f'untrained {s}')
        eu = eps_of(x0s, vu); xe_u = apply_edit(x0s, vu)[0].astype(np.float32); ce = nu.run(xe_u)['logits'].argmax(1)
        U.append(dict(seed=s, call_shares=np.bincount(cu, minlength=5).tolist(), logit_sd=float(lo.std(0).mean()), found=float(fdu.mean()),
                      eps_median=float(np.median(eu[fdu])) if fdu.any() else None, eps_q25_q75=qci(eu[fdu]) if fdu.any() else None,
                      transfer_exact=float((ce[fdu] == tu[fdu]).mean()) if fdu.any() else None, _eps=eu[fdu]))
        log(f'untrained {s}: calls {U[-1]["call_shares"]}; found {fdu.mean():.3f}; median eps {U[-1]["eps_median"]}; transfer {U[-1]["transfer_exact"]}')
        del nu
    allu = np.concatenate([u.pop('_eps') for u in U]) if U else np.array([])
    rb = np.random.default_rng(3607)
    def med_ratio():
        if not len(allu) or not len(tr_eps): return None
        bs = [np.median(rb.choice(allu, len(allu))) / np.median(rb.choice(tr_eps, len(tr_eps))) for _ in range(1000)]
        return dict(value=float(np.median(allu) / np.median(tr_eps)), ci=qci(bs))
    from scipy.stats import mannwhitneyu
    R['untrained'] = dict(jets=int(len(sub)), redraws=U, trained_found_on_same_jets=float(keep['ok'][sub].mean()), trained_eps_median=float(np.median(tr_eps)) if len(tr_eps) else None,
                          untrained_over_trained_median_eps=med_ratio(),
                          mannwhitney_p=float(mannwhitneyu(allu, tr_eps).pvalue) if len(allu) and len(tr_eps) else None)
    save()
    # ---- examples for the figure
    k = keep; ex = [int(i) for i in np.flatnonzero(k['ok'] & (k['c_f'] == W_) & (k['c_e'] == Z_))[:30]]
    np.savez_compressed(OUT / f'e36_examples{sfx}.npz', x0=k['x0'][ex].astype(np.float32), v=k['v'][ex].astype(np.float32), xe=k['xe'][ex].astype(np.float32),
                        y=k['y'][ex], tgt=k['tgt'][ex], eps=k['eps'][ex], ids=np.array(ex))
    np.savez_compressed(OUT / f'e36_edits{sfx}.npz', x0=k['x0'].astype(np.float32), v=k['v'].astype(np.float32), ok=k['ok'], prim=k['prim'], tgt=k['tgt'], c_f=k['c_f'], c_e=k['c_e'],
                        idx=k['idx'], v_random=k['RV0'].astype(np.float32), y=k['y'])
    log(f'wrote {JS}, e36_examples{sfx}.npz, e36_edits{sfx}.npz'); log('done')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', type=int, default=0); ap.add_argument('--smoke-steps', type=int, default=30); ap.add_argument('--steps', type=int, default=200); ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--kappa', type=float, default=0.05); ap.add_argument('--R', type=int, default=100); ap.add_argument('--untrained', type=int, default=5)
    ap.add_argument('--unt-n', type=int, default=400); ap.add_argument('--nboot', type=int, default=300); ap.add_argument('--nperm', type=int, default=300)
    a = ap.parse_args()
    if a.smoke: a.steps, a.R, a.untrained, a.unt_n, a.nboot, a.nperm = a.smoke_steps, 4, 1, 30, 30, 30
    main(a)
