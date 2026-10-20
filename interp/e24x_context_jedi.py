"""E24x (claims A2, A3z, A13; from E24w/E24wv): what the context-gated second-round objects S58 and S31 compute.

Structure (exact forward pass, interp.jedi_core): u_i = ReLU(W3 q(q(s_i) + q(g)) + b3) and S = scale1 * sum_i q1(u_i).
Per unit k: own_ik = W3[:, k] . q(s_i) (per particle), ctx_k = W3[:, k] . q(g) (per jet), so
  S_k ~= scale1 * sum_{i in window_k} ReLU(f_ik - theta_k(jet)),  f_ik = own_ik,  theta_k = -(ctx_k + b3_k).
Sample: fit = sample(3000, 3101) (N 15,000); test = sample(1500, 3102) minus jets also in the fit sample.
Arm 1 (naming, fit on fit, R^2 on test with 95 % jet-bootstrap CI):
  structural gated forms (exact pre-activation; additive own + ctx; theta frozen at its mean; own frozen at its slot mean),
  named gated form (f = MARS on per-particle pT, ln pT, r, rank; theta = MARS on jet observables), and references:
  (E24w) MARS <= 8 hinges on the nine pooled half-planes, (b) gradient-boosted trees on the live g units, (c) ridge on
  live g units + pairwise products, (d) theta frozen at its mean, linear in live g. theta named by MARS on (i) rotation-
  invariant jet observables (sum pT, N, pT_lead, M2, sum pT r, sum r, sum r^2, m, ...) and (ii) these plus fixed-frame
  angular count moments (the half-plane statements have zero pT weight); GBT on the same sets as an upper reference.
Arm 2 (causal, threshold): g of 3,000 test recipients overridden by g of a donor jet with matched particle content (nearest
  neighbour in the standardised slot-mean of q(s)) or by a random donor; predicted dS from theta shift (structural and
  named) vs actual dS; slope with jet-bootstrap CI; path-restricted call test: S58/S31 overridden by the actual vs the
  predicted values (other S at their originals), IIA = fraction of jets with equal calls, fraction of path call changes
  reproduced. Control: g + random direction (live g subspace) of the same norm as the matched donor shift, 200 draws x 500
  jets.
Arm 3 (causal, input): the hardest real particle outside S58's slot window (slot 0 if the window holds every slot) moved
  outward by dr = 0.1 at fixed pT and azimuth; dS with the full network vs with g frozen at its unedited value; gate effect
  = dS(full) - dS(frozen); calls (McNemar). Control edit: the softest real particle moved by the same dr.
  H0: gate effect 0 (S changes only through its own per-particle terms); H1: S changes through theta(g) as predicted.
Arm 0 (added after the smoke run showed W3[:, 58] and W3[:, 31] are non-zero only on s channels that are always 0, i.e. no
  per-particle term): every particle of 3,000 test jets replaced by a random donor's with g held at the recipient's g; max |dS|.
Theta naming by multipoles: theta_u = -(sum_k W3[k,u] ReLU(preg_k) + b3) with preg_k linear in the pooled l1 statements, i.e.
  a sum over particles of a per-particle field; preg_k fitted (linear) on count-weighted angular moments of the real particles:
  m0 (N, N(pT<11), sum ln pT, sum pT, sum r, sum r^2), + m1 (sum cos a, sin a, deta, dphi, r^2 cos a, r^2 sin a), + m2 (cos 2a, sin 2a,
  r cos 2a, r sin 2a, deta^2 - dphi^2, 2 deta dphi), + m3; passed through ReLU and W3 -> theta, S. Per-particle field: exact
  a_i through the dd kernel (1,000 fit jets), regressed on the same per-particle basis.
Untrained control: Jedi(untrained=True, seed 0/1/2): the same gating and theta-naming fits on units 58 and 31 (or, when one is
  dead, the live units with the largest context share).
Holm over 2 units x 2 causal arms.
  ./job.sh e24x-run 'python -m interp.e24x_context_jedi'  ->  $AI_RESULTS/jedi/e24x_context.json, e24x_arrays.npz
"""
import json, os, sys, time, hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
T0 = time.time()
UNITS = [58, 31]
HP = [29, 6, 40, 57, 3, 43, 30, 36, 46]
NB = 1000
SMOKE = '--smoke' in sys.argv


def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return clean(o.tolist())
    return o


_B = {}


def bidx(n, seed=24):
    if n not in _B: _B[n] = np.random.default_rng(seed).integers(0, n, (NB, n)).astype(np.int32)
    return _B[n]


def ci(v):
    v = np.asarray(v, np.float64); b = v[bidx(len(v))].mean(1)
    return dict(mean=float(v.mean()), ci=[float(np.quantile(b, .025)), float(np.quantile(b, .975))])


def boot_p(d):
    d = np.asarray(d, np.float64); b = d[bidx(len(d))].mean(1); b = b - b.mean()
    return float(min(1.0, ((np.abs(b) >= abs(d.mean())).sum() + 1) / (NB + 1)))


def r2(y, p):
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    v = ((y - y.mean()) ** 2).sum(); return float(1 - ((y - p) ** 2).sum() / v) if v > 0 else float('nan')


def r2_ci(y, p, nb=300):
    I = np.random.default_rng(7).integers(0, len(y), (nb, len(y)))
    b = [r2(y[i], p[i]) for i in I]
    return dict(r2=r2(y, p), ci=[float(np.nanquantile(b, .025)), float(np.nanquantile(b, .975))])


def slope_ci(pred, act):
    """OLS slope of actual on predicted (through the means), R^2, 95 % jet-bootstrap CI and two-sided p for slope = 0"""
    pred = np.asarray(pred, np.float64); act = np.asarray(act, np.float64)

    def sl(p, a):
        p = p - p.mean(); v = (p * p).sum(); return float((p * (a - a.mean())).sum() / v) if v > 0 else float('nan')
    I = bidx(len(pred)); b = np.array([sl(pred[i], act[i]) for i in I])
    s = sl(pred, act); bc = b - np.nanmean(b)
    p = float(min(1.0, ((np.abs(bc) >= abs(s)).sum() + 1) / (NB + 1))) if np.isfinite(s) else 1.0
    return dict(slope=s, ci=[float(np.nanquantile(b, .025)), float(np.nanquantile(b, .975))], p_slope0=p,
                r2=r2(act, pred - pred.mean() + act.mean()) if np.isfinite(s) else None,
                corr=float(np.corrcoef(pred, act)[0, 1]) if pred.std() > 0 and act.std() > 0 else None)


def mcnemar(a, b):
    a = np.asarray(a, bool); b = np.asarray(b, bool); n10 = int((a & ~b).sum()); n01 = int((~a & b).sum()); n = n10 + n01
    if n == 0: return dict(n10=0, n01=0, p=1.0)
    from scipy.stats import binomtest
    return dict(n10=n10, n01=n01, p=float(binomtest(min(n10, n01), n, 0.5).pvalue))


def holm(ps):
    ps = np.asarray(ps, np.float64); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o): run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


# ------------------------------------------------------------------ MARS-style forward selection (additive hinges)
class Mars:
    def __init__(self, names, qs=(.05, .1, .25, .5, .75, .9, .95), maxt=8, extra_knots=None):
        self.names = list(names); self.qs = qs; self.maxt = maxt; self.extra = extra_knots or {}

    def _col(self, F, t):
        k, kind, c = t
        if kind == 'lin': return F[:, k]
        if kind == 'up': return np.maximum(0, F[:, k] - c)
        return np.maximum(0, c - F[:, k])

    def fit(self, F, y, Fte=None, yte=None, sub=None):
        F = np.asarray(F, np.float64); y = np.asarray(y, np.float64)
        rows = np.arange(len(y)) if sub is None else sub
        cands = []
        for k in range(F.shape[1]):
            if F[rows, k].std() == 0: continue
            cands.append((k, 'lin', 0.0))
            knots = set(np.round(np.quantile(F[rows, k], self.qs), 6).tolist()) | set(self.extra.get(self.names[k], []))
            for c in sorted(knots):
                cands.append((k, 'up', float(c))); cands.append((k, 'dn', float(c)))
        C = {t: self._col(F[rows], t) for t in cands}
        yy = y[rows]; chosen = []; X = [np.ones(len(rows))]; path = []
        for _ in range(self.maxt):
            Q, _r = np.linalg.qr(np.column_stack(X)); res = yy - Q @ (Q.T @ yy); best = None
            for t, c in C.items():
                if t in chosen: continue
                cr = c - Q @ (Q.T @ c); v = cr @ cr
                if v < 1e-12 * max(1.0, c @ c): continue
                gain = (cr @ res) ** 2 / v
                if best is None or gain > best[0]: best = (gain, t)
            if best is None: break
            chosen.append(best[1]); X.append(C[best[1]])
            self.terms = list(chosen); self.w = np.linalg.lstsq(np.column_stack(X), yy, rcond=None)[0]
            step = dict(term=self.term_name(best[1]), r2_fit=r2(yy, np.column_stack(X) @ self.w))
            if Fte is not None: step['r2_test'] = r2(yte, self.predict(Fte))
            path.append(step)
        self.path = path; return self

    def term_name(self, t):
        k, kind, c = t; n = self.names[k]
        return n if kind == 'lin' else (f'({n} - {c:.4g})+' if kind == 'up' else f'({c:.4g} - {n})+')

    def predict(self, F):
        F = np.asarray(F, np.float64)
        return np.column_stack([np.ones(len(F))] + [self._col(F, t) for t in self.terms]) @ self.w

    def formula(self):
        return [dict(term='1', coef=float(self.w[0]))] + [dict(term=self.term_name(t), coef=float(c))
                                                           for t, c in zip(self.terms, self.w[1:])]


# ------------------------------------------------------------------ network probe
class Probe:
    """exact l1..l3 + qs1 pool arithmetic of a Jedi object, returning only what E24x needs (per-unit own / ctx / pre)"""

    def __init__(self, j, units):
        import jax, jax.numpy as jnp
        L = j.L; self.j = j; self.units = list(units)
        KB = {k: (jnp.asarray(j.KB[k][0], jnp.float32), jnp.asarray(j.KB[k][1], jnp.float32)) for k in ('l1', 'l2', 'dd', 'l3')}
        U = jnp.asarray(self.units)
        self.b3 = np.asarray(j.KB['l3'][1], np.float64)[self.units]
        self.W3 = np.asarray(j.KB['l3'][0], np.float64)
        self.scale1 = float(np.asarray(L['qs1'].scale))

        def dense(k, x):
            return L[k].activation(jnp.einsum(L[k].equation, L[k].iq(x, training=False), KB[k][0]) + KB[k][1])

        def pool(k, a):
            return jnp.sum(L[k].iq(a, training=False), axis=L[k].axes, keepdims=L[k].keepdims) * L[k].scale

        def f(xn, gov, use):
            J = xn.shape[0]
            a = dense('l1', xn); P = pool('qs', a); g = dense('dd', P)
            preg = jnp.einsum(L['dd'].equation, L['dd'].iq(P, training=False), KB['dd'][0]) + KB['dd'][1]
            if use: g = gov.reshape(g.shape)
            s = dense('l2', a)
            qs_, qg_ = L['add'].iq([s, g], training=False)
            W = KB['l3'][0][:, U]
            own = jnp.einsum('jnc,cu->jnu', qs_, W)
            ctx = jnp.einsum('jnc,cu->jnu', jnp.broadcast_to(qg_, (J, 1, qg_.shape[-1])), W)[:, 0]
            pre_all = jnp.einsum(L['l3'].equation, L['l3'].iq(qs_ + qg_, training=False), KB['l3'][0]) + KB['l3'][1]
            u = L['l3'].activation(pre_all)
            qu = L['qs1'].iq(u, training=False)
            S = jnp.sum(qu, axis=L['qs1'].axes, keepdims=L['qs1'].keepdims) * L['qs1'].scale
            return dict(P=P.reshape(J, -1), g=g.reshape(J, -1), preg=preg.reshape(J, -1), sbar=jnp.mean(qs_, axis=1), own=own, ctx=ctx,
                        pre=pre_all[..., U], qu=qu[..., U], S=S.reshape(J, -1))
        self.f0 = jax.jit(lambda xn, gov: f(xn, gov, False)); self.f1 = jax.jit(lambda xn, gov: f(xn, gov, True))

    def __call__(self, x, g=None, batch=2500):
        out = {}
        for i in range(0, len(x), batch):
            xb = self.j.inputs(np.asarray(x[i:i + batch], np.float32), self.j.n)
            if g is None: r = self.f0(xb, np.zeros((len(xb), 64), np.float32))
            else: r = self.f1(xb, np.asarray(g[i:i + batch], np.float32))
            for k, v in r.items(): out.setdefault(k, []).append(np.asarray(v, np.float64))
        return {k: np.concatenate(v) for k, v in out.items()}


# ------------------------------------------------------------------ physics features
def particle_feats(x):
    x = np.asarray(x, np.float64); pt, eta, phi = x[..., 0], x[..., 1], x[..., 2]; r = np.hypot(eta, phi)
    J, n = pt.shape; rank = np.broadcast_to(np.arange(n, dtype=np.float64), (J, n))
    return np.stack([pt, np.log(np.maximum(pt, 1e-3)), r, rank], -1), pt > 0


PNAMES = ['pT', 'ln_pT', 'r', 'rank']


def jet_obs(x, extended=False):
    x = np.asarray(x, np.float64); pt, eta, phi = x[..., 0], x[..., 1], x[..., 2]; r = np.hypot(eta, phi); real = pt > 0
    p4 = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)], -1).sum(1)
    m = np.sqrt(np.maximum(p4[:, 3] ** 2 - (p4[:, :3] ** 2).sum(1), 0.0)); M0 = pt.sum(1)
    D = dict(sum_pT=M0, ln_sum_pT=np.log(M0), N=real.sum(1).astype(float), pT_lead=pt.max(1), pT_lead_frac=pt.max(1) / M0,
             M2=(pt * r ** 2).sum(1), M2_over_M0=(pt * r ** 2).sum(1) / M0, sum_pT_r=(pt * r).sum(1),
             sum_r=(r * real).sum(1), sum_r2=(r ** 2 * real).sum(1), mean_r=(r * real).sum(1) / real.sum(1),
             m=m, m_over_pT=m / M0, sum_pT_top14=pt[:, :14].sum(1), N_pT_gt1=(pt > 1).sum(1).astype(float))
    if extended:
        e, f = eta * real, phi * real
        D.update(cnt_eta=e.sum(1), cnt_phi=f.sum(1), cnt_eta2=(e ** 2).sum(1), cnt_phi2=(f ** 2).sum(1),
                 cnt_etaphi=(e * f).sum(1), pT_eta=(pt * eta).sum(1) / M0, pT_phi=(pt * phi).sum(1) / M0,
                 pT_eta2=(pt * eta ** 2).sum(1) / M0, pT_phi2=(pt * phi ** 2).sum(1) / M0,
                 pT_etaphi=(pt * eta * phi).sum(1) / M0)
        for a in range(8):
            c, s_ = np.cos(a * np.pi / 4), np.sin(a * np.pi / 4); proj = c * eta + s_ * phi
            for t in (0.0, 0.1):
                D[f'hp{a * 45}_t{t}'] = (np.maximum(0, proj - t) * real).sum(1)
    names = list(D); return np.column_stack([D[k] for k in names]), names


def jet_hash(x):
    return np.array([hashlib.md5(np.ascontiguousarray(r).tobytes()).hexdigest() for r in np.asarray(x, np.float32)])


# ------------------------------------------------------------------ gated model
class Gate:
    """S_k ~= scale1 * sum_{i in window} ReLU(f_i - theta) for one unit (index q into Probe.units)"""

    def __init__(self, pr, D, q):
        self.q = q; self.scale = pr.scale1; self.b = pr.b3[q]
        self.window = (D['qu'][..., q] != 0).any(0)
        self.window_frac = (D['qu'][..., q] != 0).mean(0)

    def S(self, own, theta):
        return self.scale * (np.maximum(0, own - theta[:, None]) * self.window[None]).sum(1)


def fit_named(D, x, pr, q, Fte_jet, Dte, xte, ext, sub_particles=150000, rng=None):
    """named gated model for unit q: f = MARS on per-particle features (real particles in window), theta = MARS on jet obs"""
    gt = Gate(pr, D, q); own = D['own'][..., q]; theta = -(D['ctx'][:, q] + gt.b)
    PF, real = particle_feats(x); PFte, realte = particle_feats(xte)
    pad_own = float(np.median(own[~real])) if (~real).any() else 0.0
    sel = np.argwhere(real & gt.window[None])
    if len(sel) > sub_particles: sel = sel[rng.choice(len(sel), sub_particles, replace=False)]
    Fp = PF[sel[:, 0], sel[:, 1]]; yp = own[sel[:, 0], sel[:, 1]]
    mf = Mars(PNAMES, extra_knots=dict(rank=[0.5, 1.5, 2.5, 4.5, 8.5, 13.5, 20.5, 30.5, 40.5]), maxt=8).fit(Fp, yp)
    sel_te = np.argwhere(realte & gt.window[None]); ownte = Dte['own'][..., q]
    f_r2 = r2(ownte[sel_te[:, 0], sel_te[:, 1]], mf.predict(PFte[sel_te[:, 0], sel_te[:, 1]]))

    def f_named(PF_, real_):
        J, n, _ = PF_.shape; out = mf.predict(PF_.reshape(J * n, -1)).reshape(J, n)
        return np.where(real_, out, pad_own)
    Fj, names = ext
    mt = Mars(names, maxt=8).fit(Fj, theta, Fte_jet, -(Dte['ctx'][:, q] + gt.b))
    return gt, f_named, mf, mt, f_r2, pad_own



# ------------------------------------------------------------------ multipole basis (count-weighted angular moments)
def _pp_basis(pt, eta, phi):
    """per-particle basis columns (..., F) and block labels; alpha = atan2(dphi, deta), undefined at r = 0 -> angular terms 0"""
    r = np.hypot(eta, phi); ok = r > 1e-6; al = np.arctan2(phi, eta)
    c = lambda m: np.where(ok, np.cos(m * al), 0.0); sn = lambda m: np.where(ok, np.sin(m * al), 0.0)
    cols = dict(one=np.ones_like(pt), soft=(pt < 11).astype(float), ln_pT=np.log(np.maximum(pt, 1e-3)), pT=pt, r=r, r2=r ** 2,
                cos1=c(1), sin1=sn(1), deta=eta, dphi=phi, r2cos1=r ** 2 * c(1), r2sin1=r ** 2 * sn(1),
                cos2=c(2), sin2=sn(2), rcos2=r * c(2), rsin2=r * sn(2), eta2_m_phi2=eta ** 2 - phi ** 2, two_eta_phi=2 * eta * phi,
                cos3=c(3), sin3=sn(3), rcos3=r * c(3), rsin3=r * sn(3))
    blk = dict(one='m0', soft='m0', ln_pT='m0', pT='m0', r='m0', r2='m0', cos1='m1', sin1='m1', deta='m1', dphi='m1', r2cos1='m1',
               r2sin1='m1', cos2='m2', sin2='m2', rcos2='m2', rsin2='m2', eta2_m_phi2='m2', two_eta_phi='m2',
               cos3='m3', sin3='m3', rcos3='m3', rsin3='m3')
    names = list(cols); return np.stack([cols[k] for k in names], -1), names, [blk[k] for k in names]


def mp_basis(x):
    """jet-level sums over real particles of the per-particle basis ('one' -> N): (J, F), names, blocks"""
    x = np.asarray(x, np.float64); real = x[..., 0] > 0
    B, names, blk = _pp_basis(x[..., 0], x[..., 1], x[..., 2])
    return (B * real[..., None]).sum(1), ['N' if n == 'one' else 'sum_' + n for n in names], blk


LEVELS = (('m0', ('m0',)), ('m0+m1', ('m0', 'm1')), ('m0+m1+m2', ('m0', 'm1', 'm2')), ('m0..m3', ('m0', 'm1', 'm2', 'm3')))


def lsq(A, y):
    A1 = np.column_stack([np.ones(len(A)), A]); mu, sd = A1.mean(0), A1.std(0); sd[0] = 1; mu[0] = 0; sd[sd == 0] = 1
    w = np.linalg.lstsq((A1 - mu) / sd, y, rcond=None)[0]
    return lambda B: (np.column_stack([np.ones(len(B)), B]) - mu) / sd @ w, w


def theta_multipole(pr, D, Dte, x, xte, q, u, live_g):
    """theta_u = -(sum_k W3[k,u] q(g_k) + b3); g_k = ReLU(preg_k). preg_k is linear in the pooled l1 statements P, hence a sum
    over particles of a per-particle field; name it with count-weighted angular multipole sums (m0 radial/pT, m1 dipole,
    m2 quadrupole, m3), pass the fitted preg through the ReLU and W3 and compare theta and S on the test jets"""
    w = pr.W3[:, u]; ks = [int(k) for k in live_g if w[k] != 0]
    if not ks:
        c = -float(pr.b3[q]); return dict(no_live_g_input=True), (lambda Bx, name=None: np.full(len(Bx), c)), []
    B, bn, bb = mp_basis(x); Bt, _, _ = mp_basis(xte); bb = np.array(bb)
    th = -(D['ctx'][:, q] + pr.b3[q]); tht = -(Dte['ctx'][:, q] + pr.b3[q]); gt = Gate(pr, D, q)
    St = Dte['S'][:, u]; ownt = Dte['own'][..., q]
    out = dict(g_units=[dict(g=k, w3=float(w[k]), sd_g=float(D['g'][:, k].std()), frac_g_pos=float((D['g'][:, k] > 0).mean()))
                        for k in ks])
    # exact chain: theta from ReLU(preg) (no l3 / add quantizers), and preg linear in P
    th_chain = -(np.maximum(Dte['preg'][:, ks], 0) @ w[ks] + pr.b3[q])
    out['chain_exact'] = dict(theta_r2=r2(tht, th_chain), S_r2=r2_ci(St, gt.S(ownt, th_chain)))
    lp = [r2(Dte['preg'][:, k], lsq(D['P'], D['preg'][:, k])[0](Dte['P'])) for k in ks]
    out['preg_linear_in_P_r2_test'] = dict(zip([f'g{k}' for k in ks], lp))
    lev = {}; pred_fn = {}
    for name, bl in LEVELS:
        cols = np.isin(bb, bl); gp_t = []; fns = []; e = {}
        for k in ks:
            f, wk = lsq(B[:, cols], D['preg'][:, k]); fns.append(f); pt_ = f(Bt[:, cols]); gp_t.append(pt_)
            e[f'g{k}'] = dict(preg_r2=r2(Dte['preg'][:, k], pt_), g_r2=r2(Dte['g'][:, k], np.maximum(pt_, 0)))
            if name == 'm0+m1+m2':
                e[f'g{k}']['std_coefs'] = sorted([(n, float(c)) for n, c in zip([n for n, b_ in zip(bn, bb) if b_ in bl], wk[1:])],
                                                 key=lambda t: -abs(t[1]))[:8]
        tn = -(np.maximum(np.column_stack(gp_t), 0) @ w[ks] + pr.b3[q])
        lev[name] = dict(units=e, theta_r2=r2(tht, tn), S_r2=r2_ci(St, gt.S(ownt, tn)))
        pred_fn[name] = (cols, fns)
    out['levels'] = lev; out['basis_names'] = bn; out['basis_blocks'] = bb.tolist()

    def theta_mp(Bx, name='m0+m1+m2'):
        cols, fns = pred_fn[name]
        return -(np.maximum(np.column_stack([f(Bx[:, cols]) for f in fns]), 0) @ w[ks] + pr.b3[q])
    return out, theta_mp, ks


def particle_field(j, pr, x, units, ks_by_unit, n_jets=1000):
    """per-particle contribution of each real particle to preg_k (exact a_i through the dd kernel and pool scale), minus the
    padding particle's; regress the theta-combination field on the per-particle basis by multipole block"""
    Kd = np.asarray(j.KB['dd'][0], np.float64); sc = float(j.pool_scale[0])
    xs = np.asarray(x[:n_jets]); A = j.run(xs, keep=('a',), batch=250)['a']           # (J, n, 64)
    real = xs[..., 0] > 0
    Bp, names, blk = _pp_basis(xs[..., 0].astype(np.float64), xs[..., 1].astype(np.float64), xs[..., 2].astype(np.float64))
    blk = np.array(blk); res = dict(n_jets=n_jets, n_particles=int(real.sum()), kernel_shape=list(Kd.shape), pool_scale=sc)
    for u in units:
        ks = ks_by_unit[u]; w = pr.W3[ks, u]
        F = sc * A @ Kd[:, ks]                                                          # (J, n, len(ks))
        pad = F[~real].mean(0) if (~real).any() else np.zeros(len(ks))
        Th = -((F - pad) @ w)                                                           # contribution to theta per real particle
        y = Th[real]; Xb = Bp[real]; r_ = {}
        for name, bl in LEVELS:
            cols = np.isin(blk, bl); f, _ = lsq(Xb[:, cols][:, 1:], y); r_[name] = r2(y, f(Xb[:, cols][:, 1:]))
        pf = {}
        for i, k in enumerate(ks):
            yk = (F[..., i] - pad[i])[real]; pf[f'g{k}'] = {}
            for name, bl in LEVELS:
                cols = np.isin(blk, bl); f, _ = lsq(Xb[:, cols][:, 1:], yk); pf[f'g{k}'][name] = r2(yk, f(Xb[:, cols][:, 1:]))
        res[f'S{u}'] = dict(theta_field_r2_by_level=r_, g_field_r2_by_level=pf, sd_theta_field=float(y.std()),
                            pad_contribution=pad.tolist(),
                            corr_field_with={n: float(np.corrcoef(y, Xb[:, i])[0, 1]) for i, n in enumerate(names)
                                             if Xb[:, i].std() > 0})
    return res


def field_grid(pr, units, ks_by_unit, Kd, sc):
    """theta field on an (deta, dphi) grid: jets of 64 copies of one particle; per-particle field = (P(copies) - P(pad)) Kd sc"""
    e = np.linspace(-0.8, 0.8, 65); E, F_ = np.meshgrid(e, e, indexing='ij'); pts = (1.0, 3.0, 10.0, 30.0)
    xs = []
    for pt in pts:
        xg = np.zeros((E.size, 64, 3), np.float32); xg[..., 0] = pt; xg[..., 1] = E.ravel()[:, None]; xg[..., 2] = F_.ravel()[:, None]
        xs.append(xg)
    xg = np.concatenate(xs); Pg = pr(xg)['P']; Pp = pr(np.zeros((1, 64, 3), np.float32))['P']
    out = dict(eta=e, phi=e, pT=np.array(pts))
    for u in units:
        ks = ks_by_unit[u]; w = pr.W3[ks, u]
        fld = -(((Pg - Pp) @ Kd[:, ks]) * sc) @ w                                       # = 64 x per-particle field (64 copies)
        out[f'theta_field_S{u}'] = (fld / 64).reshape(len(pts), len(e), len(e))
    return out


# ------------------------------------------------------------------ main
def gating_fits(pr, D, Dte, x, xte, rng, label, R, full=True, y=None):
    """arm 1 for each unit; returns per-unit dict + models"""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    res = {}; models = {}
    FjI, nI = jet_obs(x); FjIte, _ = jet_obs(xte); FjE, nE = jet_obs(x, True); FjEte, _ = jet_obs(xte, True)
    live_g = np.flatnonzero(D['g'].std(0) > 1e-9)
    PFte, realte = particle_feats(xte)
    for q, u in enumerate(pr.units):
        S = D['S'][:, u]; Ste = Dte['S'][:, u]; gt = Gate(pr, D, q)
        r = dict(unit=f'S{u}', sd_S=float(S.std()), frac_jets_S_pos=float((S > 0).mean()),
                 window_slots=np.flatnonzero(gt.window).tolist(), n_window_slots=int(gt.window.sum()))
        if S.std() == 0:
            r['dead'] = True; res[f'S{u}'] = r; continue
        own, ctx = D['own'][..., q], D['ctx'][:, q]; ownte, ctxte = Dte['own'][..., q], Dte['ctx'][:, q]
        theta = -(ctx + gt.b); thetate = -(ctxte + gt.b)
        _, real = particle_feats(x)
        w_own = own[real & gt.window[None]]; r['sd_own_real_in_window'] = float(w_own.std())
        r['sd_own_within_jet_mean'] = float(np.nanmean(np.where(real & gt.window[None], own, np.nan).std(1)))
        r['sd_theta'] = float(theta.std()); r['mean_theta'] = float(theta.mean()); r['b3'] = float(gt.b)
        r['own_pad'] = float(np.median(own[~real])) if (~real).any() else None
        r['frac_active_particles_in_window'] = float(((own - theta[:, None]) > 0)[real & gt.window[None]].mean())
        resid = D['pre'][..., q] - own - ctx[:, None] - gt.b
        r['sd_l3_quantizer_residual'] = float(resid.std()); r['sd_pre'] = float(D['pre'][..., q].std())
        # structural forms on the test set
        exact = gt.scale * (np.maximum(0, Dte['pre'][..., q]) * gt.window[None]).sum(1)
        own_slotmean = own.mean(0)
        forms = dict(exact_pre_qs1_rounding_only=exact, additive_own_plus_ctx=gt.S(ownte, thetate),
                     theta_frozen_at_mean=gt.S(ownte, np.full(len(Ste), theta.mean())),
                     own_frozen_at_slot_mean=gt.S(np.broadcast_to(own_slotmean, ownte.shape), thetate))
        r['structural_r2_test'] = {k: r2_ci(Ste, v) for k, v in forms.items()}
        # named gated form
        gt2, f_named, mf, mtI, f_r2, pad = fit_named(D, x, pr, q, FjIte, Dte, xte, (FjI, nI), rng=rng)
        mtE = Mars(nE, maxt=8).fit(FjE, theta, FjEte, thetate)
        fte = f_named(PFte, realte)
        r['f_named'] = dict(formula=mf.formula(), path=mf.path, r2_test_per_particle=f_r2, own_pad_used=pad)
        r['theta_named_invariant'] = dict(formula=mtI.formula(), path=mtI.path, r2_test=r2(thetate, mtI.predict(FjIte)))
        r['theta_named_extended'] = dict(formula=mtE.formula(), path=mtE.path, r2_test=r2(thetate, mtE.predict(FjEte)))
        g_hgb = {}
        for nm, (A, Ate) in dict(invariant=(FjI, FjIte), extended=(FjE, FjEte)).items():
            h = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(A, theta); g_hgb[nm] = r2(thetate, h.predict(Ate))
        r['theta_gbt_r2_test'] = g_hgb
        Wl = pr.W3[live_g, u]; r['theta_linear_in_g_exact_r2_test'] = r2(thetate, -(Dte['g'][:, live_g] @ Wl + gt.b))
        named = dict(named_f_named_theta_inv=gt.S(fte, mtI.predict(FjIte)), named_f_named_theta_ext=gt.S(fte, mtE.predict(FjEte)),
                     named_f_exact_theta_inv=gt.S(ownte, mtI.predict(FjIte)), named_f_exact_theta_ext=gt.S(ownte, mtE.predict(FjEte)),
                     named_f_named_theta_exact=gt.S(fte, thetate), named_f_named_theta_mean=gt.S(fte, np.full(len(Ste), theta.mean())))
        r['named_r2_test'] = {k: r2_ci(Ste, v) for k, v in named.items()}
        # references
        ref = {}
        Z = D['P'][:, HP]; Zte = Dte['P'][:, HP]
        ref['E24w_mars_P_HP'] = Mars([f'P{h}' for h in HP], qs=(.1, .25, .5, .75, .9), maxt=8).fit(Z, S, Zte, Ste).path[-1]['r2_test']
        ref['mars_all_P'] = Mars([f'P{h}' for h in range(64)], maxt=8).fit(D['P'], S, Dte['P'], Ste).path[-1]['r2_test']
        G, Gte = D['g'][:, live_g], Dte['g'][:, live_g]
        ref['linear_live_g'] = r2(Ste, Ridge(1e-6).fit(G, S).predict(Gte))
        contrib = np.abs(pr.W3[live_g, u]) * G.std(0); top7 = live_g[np.argsort(-contrib)[:min(7, len(live_g))]]
        r['top7_g_by_weight_x_sd'] = [dict(g=int(k), w=float(pr.W3[k, u]), sd=float(D['g'][:, k].std())) for k in top7]
        G7, G7te = D['g'][:, top7], Dte['g'][:, top7]
        ref['gbt_top7_g'] = r2(Ste, HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(G7, S).predict(G7te))
        ref['gbt_live_g'] = r2(Ste, HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(G, S).predict(Gte))
        ii, jj = np.triu_indices(len(top7))
        G2, G2te = np.column_stack([G7, G7[:, ii] * G7[:, jj]]), np.column_stack([G7te, G7te[:, ii] * G7te[:, jj]])
        ref['ridge_top7_g_pairwise_products'] = r2(Ste, Ridge(1e-3).fit(G2, S).predict(G2te))
        ref['mars_invariant_obs_on_S'] = Mars(nI, maxt=8).fit(FjI, S, FjIte, Ste).path[-1]['r2_test']
        ref['mars_extended_obs_on_S'] = Mars(nE, maxt=8).fit(FjE, S, FjEte, Ste).path[-1]['r2_test']
        ref['gbt_extended_obs_on_S'] = r2(Ste, HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(FjE, S).predict(FjEte))
        r['references_r2_test'] = ref
        mp, theta_mp, ks = theta_multipole(pr, D, Dte, x, xte, q, u, live_g)
        r['theta_multipole'] = mp
        res[f'S{u}'] = r
        models[u] = dict(gate=gt, f_named=f_named, mtI=mtI, mtE=mtE, theta_mp=theta_mp, ks=ks)
        if ks: log(f'[{label}] S{u} multipole: g units {ks}; chain theta {mp["chain_exact"]["theta_r2"]:.3f} S {mp["chain_exact"]["S_r2"]["r2"]:.3f}; '
            + json.dumps({k: (round(v['theta_r2'], 3), round(v['S_r2']['r2'], 3)) for k, v in mp['levels'].items()}))
        log(f'[{label}] S{u}: sd own(in window) {r["sd_own_real_in_window"]:.3g} vs sd theta {r["sd_theta"]:.3g}; '
            f'structural {json.dumps({k: round(v["r2"], 3) for k, v in r["structural_r2_test"].items()})}; '
            f'theta inv {r["theta_named_invariant"]["r2_test"]:.3f} ext {r["theta_named_extended"]["r2_test"]:.3f} '
            f'gbt {json.dumps({k: round(v, 3) for k, v in g_hgb.items()})}; named {json.dumps({k: round(v["r2"], 3) for k, v in r["named_r2_test"].items()})}; '
            f'refs {json.dumps({k: round(v, 3) for k, v in ref.items()})}')
    return res, models, live_g


def main():
    from interp.jedi_core import sample, Jedi
    rng = np.random.default_rng(3103)
    R = dict(node='E24x', fit_sample='interp.jedi_core.sample(3000, 3101)', test_sample='interp.jedi_core.sample(1500, 3102) minus overlap')
    save = lambda: (OUT / ('e24x_smoke.json' if SMOKE else 'e24x_context.json')).write_text(json.dumps(clean(R), indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    x, y, _ = sample(*((300, 3101) if SMOKE else (3000, 3101))); xt, yt, _ = sample(*((150, 3102) if SMOKE else (1500, 3102)))
    hf = set(jet_hash(x).tolist()); keep = np.array([h not in hf for h in jet_hash(xt)])
    xt, yt = xt[keep], yt[keep]
    R['N_fit'] = len(x); R['N_test'] = len(xt); R['test_overlap_removed'] = int((~keep).sum())
    R['pt_sorted_desc_frac'] = float((np.diff(x[..., 0], axis=1) <= 0).all(1).mean())
    j = Jedi(64)
    acc = float((j.run(x)['logits'].argmax(1) == y).mean()); acct = float((j.run(xt)['logits'].argmax(1) == yt).mean())
    R['jedi_acc_fit_sample'] = acc; R['jedi_acc_test_sample'] = acct
    log(f'N fit {len(x)}, test {len(xt)} (overlap removed {R["test_overlap_removed"]}); acc fit {acc:.4f} test {acct:.4f}; '
        f'pT sorted {R["pt_sorted_desc_frac"]:.4f}')
    pr = Probe(j, UNITS)
    D = pr(x); Dt = pr(xt)
    ref = j.run(xt[:2000], keep=('S', 'g'))
    R['probe_exact'] = dict(max_abs_dS=float(np.abs(ref['S'] - Dt['S'][:2000]).max()), max_abs_dg=float(np.abs(ref['g'] - Dt['g'][:2000]).max()))
    R['S_keepdims'] = bool(j.L['qs1'].keepdims); R['scale1'] = pr.scale1
    log('probe vs jedi_core', R['probe_exact'], 'qs1 keepdims', R['S_keepdims'])
    W3 = pr.W3
    R['W3_column_stats'] = {f'u{u}': dict(n_nonzero_from_s=int((W3[:, u] != 0).sum()), b3=float(pr.b3[q]),
                                          s_units_live=int(((W3[:, u] != 0) & (D['sbar'].std(0) > 0)).sum()))
                            for q, u in enumerate(UNITS)}
    if '--untrained-only' in sys.argv:
        return untrained(x, xt, yt, rng, R, lambda: (OUT / 'e24x_untrained.json').write_text(json.dumps(clean(R), indent=1)))
    # ---------------- arm 1
    res, models, live_g = gating_fits(pr, D, Dt, x, xt, rng, 'trained', R, y=y)
    R['arm1'] = res; R['live_g_units'] = live_g.tolist(); R['dd_equation'] = str(j.L['dd'].equation); save()
    ks_by = {u: models[u]['ks'] for u in UNITS if u in models}
    R['particle_field'] = particle_field(j, pr, x, list(ks_by), ks_by, n_jets=300 if SMOKE else 1000)
    log('particle field', json.dumps(clean({u: R['particle_field'][f'S{u}']['theta_field_r2_by_level'] for u in ks_by})))
    FG = field_grid(pr, list(ks_by), ks_by, np.asarray(j.KB['dd'][0], np.float64), float(j.pool_scale[0]))
    # ---------------- arm 0: does S58 / S31 read anything but g?  all particles replaced by a random donor's, g held
    N0 = min(3000, len(xt)); r0 = rng.choice(len(xt), N0, replace=False); d0 = rng.integers(0, len(xt), N0)
    d0 = np.where(d0 == r0, (d0 + 1) % len(xt), d0)
    Ssw = j.run_jetwise(xt[d0], {'g': (np.ones(64), Dt['g'][r0])}, keep=('S',))['S']; dSw = Ssw - Dt['S'][r0]
    sdS_all = D['S'].std(0) + 1e-12; liveS = np.flatnonzero(D['S'].std(0) > 0)
    R['arm0_particle_swap_g_held'] = dict(n_jets=N0, **{f'S{u}': dict(max_abs_dS=float(np.abs(dSw[:, u]).max()),
                                                               frac_jets_changed=float((np.abs(dSw[:, u]) > 1e-9).mean())) for u in UNITS},
                                          other_live_units={f'S{k}': dict(mean_abs_dS_over_sd=float(np.abs(dSw[:, k]).mean() / sdS_all[k]),
                                                                          frac_jets_changed=float((np.abs(dSw[:, k]) > 1e-9).mean()))
                                                            for k in liveS if k not in UNITS},
                                          note='S of donor particles with the recipient g, minus the recipient S')
    log('arm0 swap (g held)', json.dumps(clean({k: v for k, v in R['arm0_particle_swap_g_held'].items() if k.startswith('S')})),
        'other live S units changed (frac jets) median', float(np.median([v['frac_jets_changed'] for v in R['arm0_particle_swap_g_held']['other_live_units'].values()])))
    save()
    # ---------------- arm 2: g interchange
    N2 = min(3000, len(xt)); rec = rng.choice(len(xt), N2, replace=False)
    sb = Dt['sbar']; mu, sd = sb.mean(0), sb.std(0) + 1e-9; Zs = (sb - mu) / sd
    don = np.empty(N2, int)
    for c in range(0, N2, 500):
        Zr = Zs[rec[c:c + 500]]; d2 = (Zr ** 2).sum(1)[:, None] + (Zs ** 2).sum(1)[None] - 2 * Zr @ Zs.T
        d2[np.arange(len(Zr)), rec[c:c + 500]] = np.inf; don[c:c + 500] = d2.argmin(1)
    don_rand = rng.integers(0, len(xt), N2); don_rand = np.where(don_rand == rec, (don_rand + 1) % len(xt), don_rand)
    xr = xt[rec]; Dr = {k: v[rec] for k, v in Dt.items()}
    base_logits = j.run(xr)['logits']; base_call = base_logits.argmax(1)
    FjI, _ = jet_obs(xt); FjE, _ = jet_obs(xt, True); PFr, realr = particle_feats(xr); Bt_, _, _ = mp_basis(xt)
    Sdims = (lambda v: v[:, None, :]) if R['S_keepdims'] else (lambda v: v)
    mS = np.zeros(64); mS[UNITS] = 1
    arm2 = {}; Rows = {}
    for dname, dd in (('matched', don), ('random', don_rand)):
        gd = Dt['g'][dd]; Da = pr(xr, g=gd)
        act = j.run_jetwise(xr, {'g': (np.ones(64), gd)}, keep=('S', 'logits'))
        A = dict(dname=dname, mean_abs_dg=float(np.abs(gd - Dr['g']).sum(1).mean()),
                 sd_matching_distance=float(np.sqrt(((Zs[rec] - Zs[dd]) ** 2).sum(1)).mean()),
                 calls_changed_full_pct=ci(100 * (act['logits'].argmax(1) != base_call)),
                 probe_vs_jetwise_max_abs_dS=float(np.abs(act['S'] - Da['S']).max()))
        for q, u in enumerate(UNITS):
            if u not in models: continue
            M = models[u]; gt = M['gate']
            dS_act = Da['S'][:, u] - Dr['S'][:, u]
            th0 = -(Dr['ctx'][:, q] + gt.b); th1 = -(Dt['ctx'][dd, q] + gt.b)
            own = Dr['own'][..., q]; f_n = M['f_named'](PFr, realr)
            pred = dict(structural=gt.S(own, th1) - gt.S(own, th0),
                        named_inv=gt.S(f_n, M['mtI'].predict(FjI[dd])) - gt.S(f_n, M['mtI'].predict(FjI[rec])),
                        named_ext=gt.S(f_n, M['mtE'].predict(FjE[dd])) - gt.S(f_n, M['mtE'].predict(FjE[rec])),
                        named_mp=gt.S(own, M['theta_mp'](Bt_[dd])) - gt.S(own, M['theta_mp'](Bt_[rec])))
            A[f'S{u}'] = dict(sd_dS_over_sd_S=float(dS_act.std() / (D['S'][:, u].std() + 1e-12)),
                              **{k: slope_ci(v, dS_act) for k, v in pred.items()})
            Rows[(dname, u)] = (dS_act, pred)
        # path-restricted call test: only S58/S31 set to actual vs predicted values (both units at once)
        Sact = Dr['S'].copy(); Sact[:, UNITS] = Da['S'][:, UNITS]
        c_act = j.run_jetwise(xr, {'S': (mS, Sdims(Sact))}, keep=('logits',))['logits'].argmax(1)
        ch_act = c_act != base_call; A['path_calls_changed_actual_pct'] = ci(100 * ch_act)
        for pk in ('structural', 'named_inv', 'named_ext', 'named_mp'):
            Sp = Dr['S'].copy()
            for q, u in enumerate(UNITS):
                if (dname, u) in Rows: Sp[:, u] = Dr['S'][:, u] + Rows[(dname, u)][1][pk]
            c_p = j.run_jetwise(xr, {'S': (mS, Sdims(Sp))}, keep=('logits',))['logits'].argmax(1)
            eq = c_p == c_act
            A[f'path_{pk}'] = dict(iia_pct=ci(100 * eq), calls_changed_pred_pct=ci(100 * (c_p != base_call)),
                                   frac_actual_changes_reproduced_pct=ci(100 * eq[ch_act]) if ch_act.sum() > 1 else None,
                                   n_actual_changes=int(ch_act.sum()), mcnemar_pred_vs_actual_change=mcnemar(c_p != base_call, ch_act))
        arm2[dname] = A
        log(f'arm2 {dname}: ' + json.dumps(clean({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk in ('mean', 'iia_pct', 'frac_actual_changes_reproduced_pct')}) for k, v in A.items() if not k.startswith('S')})))
        for u in UNITS:
            if f'S{u}' in A: log(f'   S{u}: ' + json.dumps(clean({k: (v['slope'], v['ci'], v['r2']) if isinstance(v, dict) else v for k, v in A[f'S{u}'].items()})))
    # random-direction control (live g subspace, norm of the matched donor shift), 200 draws x 500 jets
    nrd = min(500, N2); jr = np.arange(nrd); gnorm = np.linalg.norm(Dt['g'][don[jr]] - Dr['g'][jr], axis=1)
    rd = {f'S{u}': dict(dS=[], pred=[], abs_dS=[]) for u in UNITS}; rd_calls = []; rd_full = []
    xr5 = xr[jr]; base5 = base_call[jr]
    for k in range(10 if SMOKE else 200):
        v = np.zeros((nrd, 64)); vv = rng.standard_normal((nrd, len(live_g))); vv /= np.linalg.norm(vv, axis=1, keepdims=True)
        v[:, live_g] = vv * gnorm[:, None]; g2 = Dr['g'][jr] + v
        Dg = pr(xr5, g=g2)
        for q, u in enumerate(UNITS):
            if u not in models: continue
            gt = models[u]['gate']; dS = Dg['S'][:, u] - Dr['S'][jr, u]
            p = gt.S(Dr['own'][jr, :, q], -(Dg['ctx'][:, q] + gt.b)) - gt.S(Dr['own'][jr, :, q], -(Dr['ctx'][jr, q] + gt.b))
            rd[f'S{u}']['dS'].append(dS); rd[f'S{u}']['pred'].append(p)
        Sp = Dr['S'][jr].copy(); Sp[:, UNITS] = Dg['S'][:, UNITS]
        if k < (5 if SMOKE else 50):
            rd_calls.append(j.run_jetwise(xr5, {'S': (mS, Sdims(Sp))}, keep=('logits',))['logits'].argmax(1) != base5)
            rd_full.append(j.run_jetwise(xr5, {'g': (np.ones(64), g2)}, keep=('logits',))['logits'].argmax(1) != base5)
    RD = dict(n_jets=nrd, draws=200, call_draws=50, note='g + random unit direction in the live-g subspace x |g_donor - g| (matched donor), no clipping')
    for u in UNITS:
        if not rd[f'S{u}']['dS']: continue
        dS = np.stack(rd[f'S{u}']['dS'], 1); p = np.stack(rd[f'S{u}']['pred'], 1)
        RD[f'S{u}'] = dict(structural_first_draw=slope_ci(p[:, 0], dS[:, 0]),
                           structural_all_draws_r2=r2(dS.ravel(), p.ravel()),
                           mean_abs_dS_over_sd_S=ci(np.abs(dS).mean(1) / (D['S'][:, u].std() + 1e-12)),
                           matched_donor_mean_abs_dS_over_sd_S=ci(np.abs(Rows[('matched', u)][0][jr]) / (D['S'][:, u].std() + 1e-12)),
                           named_prediction='0 by construction (jet observables unchanged)')
    RD['path_calls_changed_pct'] = ci(100 * np.stack(rd_calls, 1).mean(1))
    RD['full_calls_changed_pct'] = ci(100 * np.stack(rd_full, 1).mean(1))
    arm2['random_direction'] = RD; R['arm2'] = arm2
    log('arm2 random direction', json.dumps(clean({k: v for k, v in RD.items() if not k.startswith('S')})))
    for u in UNITS:
        if f'S{u}' in RD: log(f'   S{u}', json.dumps(clean(RD[f'S{u}'])))
    save()
    # ---------------- arm 3: input edit outside S58's window, g free vs frozen
    gt58 = models[58]['gate'] if 58 in models else None
    win = gt58.window if gt58 is not None else np.ones(64, bool)
    N3 = N2; x3 = xr.copy(); real3 = x3[..., 0] > 0
    outside = real3 & ~win[None]
    has_out = outside.any(1)
    slot_hard = np.where(has_out, np.argmax(outside, 1), 0)
    nreal = real3.sum(1); slot_soft = nreal - 1
    arm3 = dict(window_S58=np.flatnonzero(win).tolist(), frac_jets_with_real_particle_outside_window=float(has_out.mean()),
                dr=0.1, n_jets=N3)

    def move(xx, slots, dr):
        xe = xx.copy(); J = np.arange(len(xe)); e, f = xe[J, slots, 1].astype(np.float64), xe[J, slots, 2].astype(np.float64)
        rr = np.hypot(e, f); ang = np.where(rr > 1e-6, np.arctan2(f, e), rng.uniform(-np.pi, np.pi, len(xe)))
        xe[J, slots, 1] = (rr + dr) * np.cos(ang); xe[J, slots, 2] = (rr + dr) * np.sin(ang); return xe
    ps = []
    for ename, slots in (('hard_outside_window', slot_hard), ('soft_control', slot_soft)):
        xe = move(x3, slots, 0.1); De = pr(xe); Dfz = pr(xe, g=Dr['g'])
        full = j.run(xe)['logits'].argmax(1); frz = j.run_jetwise(xe, {'g': (np.ones(64), Dr['g'])}, keep=('logits',))['logits'].argmax(1)
        ch_full, ch_frz = full != base_call, frz != base_call
        E = dict(slot_mean=float(slots.mean()), slot0_frac=float((slots == 0).mean()),
                 mean_abs_dg=float(np.abs(De['g'] - Dr['g']).sum(1).mean()),
                 calls_changed_full_pct=ci(100 * ch_full), calls_changed_g_frozen_pct=ci(100 * ch_frz),
                 mcnemar_full_vs_frozen=mcnemar(ch_full, ch_frz), diff_calls_pct=ci(100 * (ch_full.astype(float) - ch_frz)))
        FjIe, _ = jet_obs(xe); FjEe, _ = jet_obs(xe, True); PFe, reale = particle_feats(xe)
        FjIr, FjEr = FjI[rec], FjE[rec]
        for q, u in enumerate(UNITS):
            if u not in models: continue
            M = models[u]; gt = M['gate']; sdS = D['S'][:, u].std()
            dfull = De['S'][:, u] - Dr['S'][:, u]; dfrz = Dfz['S'][:, u] - Dr['S'][:, u]; gate = dfull - dfrz
            f0 = M['f_named'](PFr, realr); f1 = M['f_named'](PFe, reale)
            pg = dict(structural=gt.S(De['own'][..., q], -(De['ctx'][:, q] + gt.b)) - gt.S(De['own'][..., q], -(Dr['ctx'][:, q] + gt.b)),
                      named_inv=gt.S(f1, M['mtI'].predict(FjIe)) - gt.S(f1, M['mtI'].predict(FjIr)),
                      named_ext=gt.S(f1, M['mtE'].predict(FjEe)) - gt.S(f1, M['mtE'].predict(FjEr)),
                      named_mp=gt.S(De['own'][..., q], M['theta_mp'](mp_basis(xe)[0])) - gt.S(De['own'][..., q], M['theta_mp'](Bt_[rec])))
            p_gate = boot_p(gate / sdS)
            E[f'S{u}'] = dict(dS_full_over_sd=ci(dfull / sdS), abs_dS_full_over_sd=ci(np.abs(dfull) / sdS),
                              dS_g_frozen_over_sd=ci(dfrz / sdS), abs_dS_g_frozen_over_sd=ci(np.abs(dfrz) / sdS),
                              gate_effect_over_sd=ci(gate / sdS), abs_gate_effect_over_sd=ci(np.abs(gate) / sdS),
                              p_gate_zero=p_gate, frac_jets_gate_nonzero=float((np.abs(gate) > 1e-9).mean()),
                              gate_prediction={k: slope_ci(v, gate) for k, v in pg.items()})
            if ename == 'hard_outside_window': ps.append((f'arm3_S{u}_named_mp_slope', E[f'S{u}']['gate_prediction']['named_mp']['p_slope0']))
        arm3[ename] = E
        log(f'arm3 {ename}: ' + json.dumps(clean({k: (v['mean'] if isinstance(v, dict) and 'mean' in v else v) for k, v in E.items() if not k.startswith('S')})))
        for u in UNITS:
            if f'S{u}' in E:
                e_ = E[f'S{u}']; log(f'   S{u}: full {e_["dS_full_over_sd"]["mean"]:.4f} |.| {e_["abs_dS_full_over_sd"]["mean"]:.4f}; frozen {e_["dS_g_frozen_over_sd"]["mean"]:.4f} |.| {e_["abs_dS_g_frozen_over_sd"]["mean"]:.4f}; '
                                     f'gate {e_["gate_effect_over_sd"]} p {e_["p_gate_zero"]:.4g}; pred ' + json.dumps(clean({k: (v['slope'], v['r2']) for k, v in e_['gate_prediction'].items()})))
    R['arm3'] = arm3
    for u in UNITS:
        if f'S{u}' in arm2['matched']: ps.append((f'arm2_S{u}_named_mp_slope', arm2['matched'][f'S{u}']['named_mp']['p_slope0']))
    adj = holm([p for _, p in ps]); R['holm'] = {k: dict(p=p, holm=float(a)) for (k, p), a in zip(ps, adj)}
    log('holm', json.dumps(clean(R['holm']))); save()
    # arrays for the figure
    np.savez_compressed(OUT / ('e24x_smoke.npz' if SMOKE else 'e24x_arrays.npz'), **{f'{d}_S{u}_act': Rows[(d, u)][0] for (d, u) in Rows},
                        **{f'{d}_S{u}_pred_{k}': v for (d, u) in Rows for k, v in Rows[(d, u)][1].items()},
                        theta58_test=-(Dt['ctx'][:, 0] + pr.b3[0]), theta31_test=-(Dt['ctx'][:, 1] + pr.b3[1]),
                        S58_test=Dt['S'][:, 58], S31_test=Dt['S'][:, 31],
                        **{f'grid_{k}': np.asarray(v) for k, v in FG.items()},
                        **{f'mp_theta_S{u}_test': models[u]['theta_mp'](mp_basis(xt)[0]) for u in models})
    untrained(x, xt, yt, rng, R, save)


def untrained(x, xt, yt, rng, R, save):
    from interp.jedi_core import Jedi
    R['untrained'] = {}
    nu, nut = 6000, 3000
    for seed in ((0,) if SMOKE else (0, 1, 2)):
        ju = Jedi(64, untrained=True, seed=seed)
        Dq = Probe(ju, list(range(64)))
        Du = Dq(x[:1500])
        sdS = Du['S'].std(0)
        ctx_share = Du['ctx'].std(0) / (Du['own'].reshape(-1, 64).std(0) + Du['ctx'].std(0) + 1e-12)
        pick = [u for u in UNITS if sdS[u] > 1e-9]
        cand = [int(k) for k in np.argsort(-ctx_share) if sdS[k] > 1e-9 and k not in pick]
        pick = (pick + cand)[:2]
        pu = Probe(ju, pick); Du = pu(x[:nu]); Dut = pu(xt[:nut])
        res_u, _, _ = gating_fits(pu, Du, Dut, x[:nu], xt[:nut], rng, f'untrained s{seed}', R)
        R['untrained'][f'seed{seed}'] = dict(units=pick, ctx_share={f'u{k}': float(ctx_share[k]) for k in pick},
                                             acc=float((ju.run(xt[:nut])['logits'].argmax(1) == yt[:nut]).mean()), fits=res_u)
        save(); del ju, Dq, pu
    log('done'); save()


if __name__ == '__main__':
    main()
