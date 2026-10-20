"""X10 (claims A4, A9, B2m; review weakness 3): distributed alignment search (DAS) for mass / class variables at S, h4, h5
of JEDI-linear (3-feature, n64) against random orthonormal subspaces of the same dimension.

One job per network (all three stages share the forward passes of that network):
  ./job.sh x10-trained 'python -m interp.x10_das_jedi --net trained'       (DAS, random null, permuted-label DAS, calibration)
  ./job.sh x10-u0      'python -m interp.x10_das_jedi --net untrained0'    (untrained=True seed 0; also untrained1, untrained2)
  ./job.sh x10-pfn     'python -m interp.x10_das_jedi --net pfn'           (X9 PFN (64,64,32) s0, pooled latent L, 32 dims)
  ./job.sh x10-sum     'python -m interp.x10_das_jedi --summary'           (Holm over 4 variables x 3 stages x 3 k; figure data)

Debug 1 (after the first runs, 2026-10-19): the DAS training tail for JEDI-linear is the exact fixed-point tail with a
  straight-through gradient through each input quantizer (the float copy disagreed with the exact tail on interchanged states);
  untrained networks call argmax(logits - mean logits on F) because their raw call is one class for every jet. PFN unchanged.
Debug 2 (2026-10-19): the next layer's input quantizer is fixed point and wraps modulo 2^i, so an interchanged state outside
  its range [-k 2^i, 2^i - 2^-f] wraps to an unrelated value. Every interchanged / patched state (DAS, random null, calibration)
  is clipped per unit to that range, widened to include the base value (unedited states are never changed).

FROZEN HEADER (written before any run)
 Samples. Fit F = sample(per_class=4000, seed=0) (20,000 jets): DAS training pairs. Test T = sample(per_class=4000, seed=17)
   minus the jets also in F. Pairs: training 8,000 ordered (base, source) pairs from F (seed 2), last 20 % held out for model
   selection; test 5,000 ordered pairs from T (pair seed 1, base != source) ("random arm"); on-manifold arm: 5,000 pairs from
   T with the source's ΣpT of the 64 stored particles (M0) within ±5 % of the base's (seed 3).
 High-level variables V (about the stored jet axis; real particles only, padding never edited):
   M0 = Σ pT_i;  M2 = Σ pT_i r_i^2, r_i^2 = Δη_i^2 + Δφ_i^2;  D = ½ log(M0 M2);  WZ = l_W − l_Z (d7 logit margin).
 H's counterfactual for a pair (base b, source s) = the network's own call on b with V set to V(s) by an input edit:
   do(M0): pT × c, angles × c^-1/2, c = M0_s / M0_b                    (M2 unchanged)
   do(M2): angles × k, k = sqrt(M2_s / M2_b)                           (M0 unchanged)
   do(D):  pT × c, c = sqrt(M0_s M2_s / (M0_b M2_b))                   (M0, M2 both × c; = exact m × c for massless particles)
   do(WZ): base logits with (l_W, l_Z) moved to their base mean ± margin_s / 2 (no input edit).
   The exact network (fixed point, interp.jedi_core) runs the edited jets, so the full-replacement counterfactual has no fit error.
 DAS. For stage L in {S (64), h4 (64), h5 (32)} and each V and k in {1, 2, 4}: R = qr(W), W in R^{d x k}; interchange
   h' = h_b + (h_s − h_b) R R^T; loss = weighted cross-entropy of a float copy of the downstream layers (fused kernels, no
   quantizers) on h' against H's counterfactual call; pairs whose counterfactual call differs from the base call carry half the
   weight. Adam (lr 0.01), 800 full-batch steps, model selection every 50 steps on the held-out training pairs (float tail).
   Evaluation uses the exact fixed-point downstream layers (checked against run_jetwise).
 Metrics on test pairs: IIA = fraction of pairs where the intervened network's call equals H's counterfactual call;
   effective IIA (primary) = the same on the pairs where H's counterfactual call differs from the base call (E_H; the same pair
   set for DAS and every control, so comparisons are paired); E22-style effective IIA on the pairs where the intervention changes
   the network's call (E_L) also reported. 95 % CI: 1000 bootstrap resamples of test pairs (each pair has its own base jet).
 Controls. (i) 1000 random orthonormal k-dim subspaces (raw stage coordinates) per stage and k; p_rot = (1 + #{random >= DAS}) /
   1001. (ii) DAS trained with the counterfactual labels of a deranged pair (mismatched source), evaluated on true labels.
   (iii) the same DAS on the untrained network (jedi_core untrained=True, seeds 0-2), labels from its own edited calls; 200
   random subspaces each. (iv) the X9 PFN pooled latent L (32 dims), same code; 1000 random subspaces.
 Transfer: the subspace learned for V evaluated against the counterfactuals of every other V' (same test pairs).
 Calibration (trained, V in {M0, M2, D}, k in {1, 2}): 5,000 test jets (random rows of T, seed 4). Real edits: 0.06 rad
   translation in a random direction per jet (m unchanged; ½Δlog M2 > 0) and angle scaling by exp(mean ½Δlog M2 of the
   translation) (a real mass change of the predicted size). Patch: h = h_c + (h_edit − h_c) R R^T at the stage, exact tail.
   Read-out: ridge of log m on z (fitted on F, 2/3 fit, 1/3 dev). Metrics: Δlog m̂ (patched) / Δlog m̂ (full edit); W → Z among
   true-W jets; null: 1000 random subspaces (p = rank); 1000 jet-bootstrap CIs.
 Tests. H0: DAS effective IIA = random-subspace effective IIA (p_rot >= 0.05); H1: higher. Holm over 4 V x 3 stages x 3 k = 36
   (--summary). DAS vs permuted-label DAS: paired bootstrap difference.
 Predictions. Thesis: D aligned at h4/h5 with effective IIA >= 0.8 and p_rot < 0.01; M0, M2 aligned already at S (k = 1-2);
   untrained at chance. Alternative (distributed variables, no causal subspace): DAS within the random band at all k <= 4, or D
   only at S with the same value as M2 (D never formed as a separable variable).
Writes $AI_RESULTS/jedi/x10_<net>.json (partial after each stage) and x10_summary.json.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'
T0 = time.time(); W_, Z_ = 2, 3
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
VARS = ('M0', 'M2', 'D', 'WZ'); EDITV = ('M0', 'M2', 'D'); KS = (1, 2, 4)
STAGES = ('S', 'h4', 'h5'); WID = dict(S=64, h4=64, h5=32, z=16, L=32)


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return clean(o.tolist())
    return o


def holm(ps):
    ps = np.asarray(ps, np.float64); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o): run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


# ------------------------------------------------------------------------------------------------ physics
def moments(x):
    x = np.asarray(x, np.float64); pt = x[..., 0]
    return pt.sum(1), (pt * (x[..., 1] ** 2 + x[..., 2] ** 2)).sum(1)


def mass(x):
    x = np.asarray(x, np.float64); pt, eta, phi = x[..., 0], x[..., 1], x[..., 2]
    p4 = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)], -1).sum(1)
    return np.sqrt(np.maximum(p4[:, 3] ** 2 - (p4[:, :3] ** 2).sum(1), 0.0))


def scale(x, c, a):
    """pT × c, angles × a (per jet) on real particles; padding untouched"""
    e = np.array(x, np.float64); rl = e[..., 0] > 0; c = np.asarray(c, np.float64)[:, None]; a = np.asarray(a, np.float64)[:, None]
    e[..., 0] = np.where(rl, e[..., 0] * c, e[..., 0])
    for j in (1, 2): e[..., j] = np.where(rl, e[..., j] * a, e[..., j])
    return e.astype(np.float32)


def do_edit(xb, V, mb, ms):
    (M0b, M2b), (M0s, M2s) = mb, ms; one = np.ones(len(xb))
    if V == 'M0': c = M0s / M0b; return scale(xb, c, c ** -0.5)
    if V == 'M2': return scale(xb, one, np.sqrt(M2s / M2b))
    if V == 'D': return scale(xb, np.sqrt(M0s * M2s / (M0b * M2b)), one)
    raise ValueError(V)


def translate(x, d, th):
    e = np.array(x, np.float32); rl = e[..., 0] > 0
    e[..., 1] += (d * np.cos(th))[:, None].astype(np.float32) * rl; e[..., 2] += (d * np.sin(th))[:, None].astype(np.float32) * rl
    return e


def wz_call(lo_b, lo_s):
    lo = np.array(lo_b, np.float64); mid = (lo[:, W_] + lo[:, Z_]) / 2; mg = lo_s[:, W_] - lo_s[:, Z_]
    lo[:, W_] = mid + mg / 2; lo[:, Z_] = mid - mg / 2; return lo.argmax(1)


def indices(per_class, seed):
    from jetdistill.config import RESULTS
    y = np.asarray(np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')['y']); rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per_class, replace=False) for c in range(5)]))


class RidgeR:
    def __init__(self, X, t, fit, dev):
        al = np.concatenate([fit, dev]); self.mu = X[al].mean(0); self.sd = X[al].std(0); self.sd[self.sd < 1e-9] = np.inf
        Zs = (X - self.mu) / self.sd; best = None
        for a in 10.0 ** np.arange(-4, 4.1, 0.5):
            w, b = self._fit(Zs[fit], t[fit], a); e = ((Zs[dev] @ w + b - t[dev]) ** 2).mean()
            if best is None or e < best[0]: best = (e, a)
        self.w, self.b = self._fit(Zs[al], t[al], best[1]); self.alpha = best[1]

    @staticmethod
    def _fit(Z, t, a):
        mz, mt = Z.mean(0), t.mean(); Zc = Z - mz
        w = np.linalg.solve(Zc.T @ Zc + a * len(Z) * np.eye(Z.shape[1]), Zc.T @ (t - mt)); return w, mt - mz @ w

    def __call__(self, X): return ((X - self.mu) / self.sd) @ self.w + self.b


# ------------------------------------------------------------------------------------------------ networks
class JNet:
    """JEDI-linear: exact fixed-point tail from S / h4 / h5 (logits, z) and a float differentiable copy (jax)"""

    def __init__(self, untrained=False, seed=0):
        from interp.jedi_core import Jedi
        import jax, jax.numpy as jnp
        self.net = Jedi(64, untrained=untrained, seed=seed); L = self.net.L; KB = {}
        for k in ('d4', 'd5', 'd6', 'd7'):
            K, b = L[k].get_fused_qkernel_and_qbias(False, L[k].moving_mean, L[k].moving_variance); KB[k] = (jnp.asarray(K), jnp.asarray(b))
        self.stages = STAGES

        def dense(k, v, act=True):
            z = jnp.einsum(L[k].equation, L[k].iq(v, training=False), KB[k][0]) + KB[k][1]
            return L[k].activation(z) if act and L[k].activation is not None else z

        def from_h5(h5):
            z = dense('d6', h5, act=False); return dense('d7', jax.nn.relu(z)).reshape(h5.shape[0], -1), z.reshape(h5.shape[0], -1)
        def from_h4(h4): return from_h5(dense('d5', h4))
        def from_S(S): return from_h4(dense('d4', S))
        self._ex = dict(S=jax.jit(from_S), h4=jax.jit(from_h4), h5=jax.jit(from_h5)); self.shp = None; self.off = np.zeros(5)

        # debug 1 (2026-10-19): training tail = the exact fixed-point tail with a straight-through gradient through each input
        # quantizer (the float copy agreed on 99 % of unedited calls but not on interchanged states, which the quantizers clip)
        def dste(k, v, act=True):
            q = L[k].iq(v, training=False); q = v + jax.lax.stop_gradient(q - v)
            z = jnp.einsum(L[k].equation, q, KB[k][0]) + KB[k][1]
            return L[k].activation(z) if act and L[k].activation is not None else z

        def s5(h5): return dste('d7', jax.nn.relu(dste('d6', h5, act=False))).reshape(h5.shape[0], -1)
        def s4(h4): return s5(dste('d5', h4))
        def sS(S): return s4(dste('d4', S))
        self.ste = dict(S=sS, h4=s4, h5=s5)
        self.rng_ = {}
        for st, k in (('S', 'd4'), ('h4', 'd5'), ('h5', 'd6')):
            q = L[k].iq.quantizer; kk, ii, ff = (np.asarray(getattr(q, a), np.float64).ravel() for a in ('k', 'i', 'f'))
            d = WID[st]; kk, ii, ff = (np.broadcast_to(v, (d,)) if v.size == 1 else v for v in (kk, ii, ff))
            self.rng_[st] = (-kk * 2.0 ** ii, 2.0 ** ii - 2.0 ** -ff)
        # float copy: layer list after each stage, (K (in, out), b, relu)
        Kf = {k: (np.asarray(v[0], np.float32).reshape(-1, np.asarray(v[0]).shape[-1]), np.asarray(v[1], np.float32).reshape(-1)) for k, v in self.net.KB.items()}
        seq = [('d4', True), ('d5', True), ('d6', True), ('d7', False)]
        self.flayers = dict(S=[(Kf[k][0], Kf[k][1], r) for k, r in seq], h4=[(Kf[k][0], Kf[k][1], r) for k, r in seq[1:]],
                            h5=[(Kf[k][0], Kf[k][1], r) for k, r in seq[2:]])

    def reps(self, x, keep=('S', 'h4', 'h5', 'z', 'logits')):
        return self.net.run(x, keep=keep)

    def calls(self, x):
        return (self.net.run(x, keep=('logits',))['logits'] - self.off).argmax(1)

    def tail(self, stage):
        import jax.numpy as jnp
        f, off = self.ste[stage], jnp.asarray(self.off, jnp.float32)
        return lambda h: f(h) - off

    def exact(self, stage, h, batch=5000):
        lo, zz = [], []
        for i in range(0, len(h), batch):
            hb = np.asarray(h[i:i + batch], np.float32)
            if self.shp == 'b': hb = hb.reshape(len(hb), 1, -1)
            a, b = self._ex[stage](hb); lo.append(np.asarray(a, np.float64) - self.off); zz.append(np.asarray(b, np.float64))
        return np.concatenate(lo), np.concatenate(zz)

    def check(self, x, R):
        """exact tail reproduces run() at every entry stage and an S / h5 override of run_jetwise"""
        for shp in ('a', 'b'):
            self.shp = shp
            try:
                d0 = max(float(np.abs(self.exact(s, R[s])[0] - R['logits']).max()) for s in STAGES)
                rr = np.random.default_rng(5).permutation(len(x))[:1000]; out = {}
                for s in ('S', 'h5'):
                    v = R[s][rr].copy(); v[:, :3] = R[s][rr[::-1], :3]
                    lo = self.net.run_jetwise(x[rr], {s: (np.ones(WID[s]), v)})['logits'] - self.off; out[s] = float(np.abs(self.exact(s, v)[0] - lo).max())
                log(f'exact tail shape {shp}: max |dlogit| base {d0:.2e}, S override {out["S"]:.2e}, h5 override {out["h5"]:.2e}')
                if max(d0, *out.values()) < 1e-4: return dict(shape=shp, base=d0, **out)
            except Exception as e:  # noqa: BLE001
                log(f'tail shape {shp} failed: {e!r}')
        raise SystemExit('exact tail does not reproduce the network')


class PNet:
    """X9 PFN (64,64,32) seed 0: pooled latent L; the tail (F layers) is float, so exact = float copy"""

    def __init__(self, name='pfn_64-64-32_s0'):
        from interp.x9_efn_pfn import Flow
        self.M = Flow(name); self.stages = ('L',)
        self.flayers = dict(L=[(K.astype(np.float32), b.astype(np.float32), True) for K, b in self.M.F] +
                            [(self.M.out[0].astype(np.float32), self.M.out[1].astype(np.float32), False)])

    def reps(self, x, keep=None):
        r = self.M.run(x, keep=('L', 'logits')); return dict(L=r['L'], logits=r['logits'])

    def calls(self, x):
        return self.M.run(x, keep=('logits',))['logits'].argmax(1)

    def exact(self, stage, h, batch=None):
        return self.M.tail(np.asarray(h, np.float64)), None

    def tail(self, stage):
        return make_ftail(self.flayers[stage])

    rng_ = dict(L=(np.full(32, -np.inf), np.full(32, np.inf)))

    def check(self, x, R):
        d = float(np.abs(self.exact('L', R['L'])[0] - R['logits']).max()); log(f'PFN tail check {d:.2e}'); return dict(base=d)


def make_ftail(layers):
    import jax, jax.numpy as jnp
    Ls = [(jnp.asarray(K), jnp.asarray(b), r) for K, b, r in layers]

    def f(h):
        for K, b, r in Ls:
            h = h @ K + b
            if r: h = jax.nn.relu(h)
        return h
    return f


def proj(rg, hb, h):
    """clip h per unit to the next layer's representable range, widened to include the base value"""
    lo, hi = rg; return np.clip(h, np.minimum(lo, hb), np.maximum(hi, hb))


def das_fit(ftail, hb, hs, lab, basec, k, seed, steps=800, lr=0.01, rg=None):
    """learn an orthonormal k-dim subspace; returns (R (d, k), info)"""
    import jax, jax.numpy as jnp
    n = len(lab); nv = n // 5; tr = np.arange(n - nv); va = np.arange(n - nv, n)
    chg = lab != basec; w = np.where(chg, 0.5 / max(chg[tr].sum(), 1), 0.5 / max((~chg[tr]).sum(), 1)).astype(np.float32)
    Hb, Hs, Y, Wt = (jnp.asarray(a) for a in (hb[tr].astype(np.float32), hs[tr].astype(np.float32), lab[tr], w[tr]))
    Vb, Vs = jnp.asarray(hb[va].astype(np.float32)), jnp.asarray(hs[va].astype(np.float32))
    lo_, hi_ = (jnp.asarray(np.nan_to_num(v, posinf=3e38, neginf=-3e38).astype(np.float32)) for v in rg)
    cl = lambda b_, h_: jnp.clip(h_, jnp.minimum(lo_, b_), jnp.maximum(hi_, b_))

    def proj(Wm):
        Q, _ = jnp.linalg.qr(Wm); return Q

    def loss(Wm):
        Q = proj(Wm); h = cl(Hb, Hb + ((Hs - Hb) @ Q) @ Q.T); lo = ftail(h)
        lp = jax.nn.log_softmax(lo); return -(Wt * lp[jnp.arange(len(Y)), Y]).sum()
    vg = jax.jit(jax.value_and_grad(loss))

    @jax.jit
    def vcall(Wm):
        Q = proj(Wm); return ftail(cl(Vb, Vb + ((Vs - Vb) @ Q) @ Q.T)).argmax(1)
    rng = np.random.default_rng(seed); Wm = jnp.asarray(rng.normal(size=(hb.shape[1], k)).astype(np.float32))
    m = jnp.zeros_like(Wm); v = jnp.zeros_like(Wm); b1, b2 = 0.9, 0.999; best = (-1, None, 0); hist = []
    chv = chg[va]
    for t in range(1, steps + 1):
        L, g = vg(Wm); m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        Wm = Wm - lr * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + 1e-8)
        if t % 50 == 0 or t == steps:
            c = np.asarray(vcall(Wm)); ok = c == lab[va]
            sc = 0.5 * ok[chv].mean() + 0.5 * ok[~chv].mean() if chv.any() and (~chv).any() else ok.mean()
            hist.append((t, float(L), float(sc)))
            if sc > best[0]: best = (float(sc), np.asarray(proj(Wm), np.float64), t)
    return best[1], dict(val_balanced_agreement=best[0], best_step=best[2], loss_first=hist[0][1], loss_last=hist[-1][1])


def eval_sub(net, stage, hb, hs, Q, labs, basec):
    h = proj(net.rng_[stage], hb, hb + ((hs - hb) @ Q) @ Q.T); c = net.exact(stage, h)[0].argmax(1); return c


def scores(c, lab, basec):
    EH = lab != basec; EL = c != basec; ok = c == lab
    return dict(iia=float(ok.mean()), eff=float(ok[EH].mean()) if EH.sum() >= 20 else None, n_eff=int(EH.sum()),
                effL=float(ok[EL].mean()) if EL.sum() >= 20 else None, n_effL=int(EL.sum()))


def boot_eff(c, lab, basec, B=1000, seed=7):
    rng = np.random.default_rng(seed); ok = (c == lab).astype(np.float64); EH = (lab != basec).astype(np.float64); n = len(c); v = []
    for _ in range(B):
        i = rng.integers(0, n, n); d = EH[i].sum()
        if d >= 20: v.append((ok[i] * EH[i]).sum() / d)
    return [float(np.quantile(v, .025)), float(np.quantile(v, .975))] if v else [None, None]


def boot_diff(c1, c2, lab, basec, B=1000, seed=8):
    rng = np.random.default_rng(seed); EH = (lab != basec).astype(np.float64); a = (c1 == lab) * EH; b = (c2 == lab) * EH; n = len(lab); v = []
    for _ in range(B):
        i = rng.integers(0, n, n); d = EH[i].sum()
        if d >= 20: v.append((a[i].sum() - b[i].sum()) / d)
    v = np.asarray(v)
    if not len(v): return None
    return dict(diff=float((a.sum() - b.sum()) / max(EH.sum(), 1)), ci=[float(np.quantile(v, .025)), float(np.quantile(v, .975))],
                p=float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()) + 1 / (len(v) + 1))))


def rand_Q(d, k, rng):
    Q, _ = np.linalg.qr(rng.normal(size=(d, k))); return Q


# ------------------------------------------------------------------------------------------------ main
def main(a):
    sm = a.smoke; tag = a.net + ('_smoke' if sm else ''); JS = OUT / f'x10_{tag}.json'; OUT.mkdir(parents=True, exist_ok=True)
    R = dict(net=a.net, smoke=sm, header='see module docstring (frozen)')

    def save(): JS.write_text(json.dumps(clean(R), indent=1))
    from interp.jedi_core import sample
    pc = 300 if sm else 4000
    xF, yF, _ = sample(pc, 0); xT, yT, _ = sample(pc, 17)
    iF, iT = indices(pc, 0), indices(pc, 17); keep = ~np.isin(iT, iF); xT, yT = xT[keep], yT[keep]
    NF, NT = len(yF), len(yT); log(f'F {NF} jets; T {NT} jets (removed {int((~keep).sum())} shared with F)')
    R['N'] = dict(F=NF, T=NT, T_removed=int((~keep).sum()))
    mF, mT = moments(xF), moments(xT)
    npair = 800 if sm else 8000; ntest = 600 if sm else 5000
    r2 = np.random.default_rng(2); fb = r2.integers(0, NF, npair); fs = r2.integers(0, NF, npair)
    while (fb == fs).any(): fs[fb == fs] = r2.integers(0, NF, (fb == fs).sum())
    r1 = np.random.default_rng(1); tb = r1.integers(0, NT, ntest); ts = r1.integers(0, NT, ntest)
    while (tb == ts).any(): ts[tb == ts] = r1.integers(0, NT, (tb == ts).sum())
    # on-manifold: source M0 within ±5 % of the base's
    r3 = np.random.default_rng(3); o = np.argsort(mT[0]); srt = mT[0][o]; mb_, ms_ = [], []
    while len(mb_) < ntest:
        b = int(r3.integers(0, NT)); lo_, hi_ = np.searchsorted(srt, mT[0][b] * 0.95), np.searchsorted(srt, mT[0][b] * 1.05)
        cand = o[lo_:hi_]; cand = cand[cand != b]
        if len(cand): mb_.append(b); ms_.append(int(r3.choice(cand)))
    mb, ms = np.array(mb_), np.array(ms_)
    pairs = dict(train=(xF, mF, fb, fs), rand=(xT, mT, tb, ts), mani=(xT, mT, mb, ms))
    R['pairs'] = {k: dict(n=len(v[2]), log_ratio_M0_sd=float(np.std(np.log(v[1][0][v[3]] / v[1][0][v[2]]))),
                          log_ratio_M2_sd=float(np.std(np.log(v[1][1][v[3]] / v[1][1][v[2]]))),
                          frac_angle_scale_gt3=float((np.sqrt(v[1][1][v[3]] / v[1][1][v[2]]) > 3).mean())) for k, v in pairs.items()}
    log('pairs', R['pairs'])

    # network
    if a.net == 'trained': net = JNet()
    elif a.net.startswith('untrained'): net = JNet(untrained=True, seed=int(a.net[-1]))
    elif a.net == 'pfn': net = PNet()
    else: raise SystemExit(a.net)
    RF = net.reps(xF); RT = net.reps(xT)
    if a.net.startswith('untrained'):  # debug 1: untrained nets give one call for every jet; calls = argmax of centred logits
        net.off = RF['logits'].mean(0); RF['logits'] = RF['logits'] - net.off; RT['logits'] = RT['logits'] - net.off
        R['logit_offset'] = net.off; log(f'untrained: logits centred on the F mean {net.off}')
    cF = RF['logits'].argmax(1); cT = RT['logits'].argmax(1)
    acc = float((cT == yT).mean()); R['acc_T'] = acc; R['acc_F'] = float((cF == yF).mean())
    R['call_dist_T'] = np.bincount(cT, minlength=5).tolist()
    log(f'accuracy T {acc:.4f} F {R["acc_F"]:.4f}; calls {R["call_dist_T"]}')
    R['tail_check'] = net.check(xT[:2000] if not sm else xT, {k: v[:2000] for k, v in RT.items()}); save()

    # H counterfactual labels
    lab = {}
    for arm, (x, m, b, s) in pairs.items():
        Rb = RF if arm == 'train' else RT; base = Rb['logits'].argmax(1)[b]; lab[arm] = dict(base=base)
        for V in EDITV:
            xe = do_edit(x[b], V, (m[0][b], m[1][b]), (m[0][s], m[1][s])); lab[arm][V] = net.calls(xe)
        lab[arm]['WZ'] = wz_call(Rb['logits'][b], Rb['logits'][s])
        log(f'labels {arm}: fraction counterfactual call != base: ' + ', '.join(f'{V} {(lab[arm][V] != base).mean():.3f}' for V in VARS))
    R['cf_change_frac'] = {arm: {V: float((lab[arm][V] != lab[arm]['base']).mean()) for V in VARS} for arm in lab}
    # float copy agreement
    R['float_tail'] = {}
    for st in net.stages:
        fc = np.asarray(make_ftail(net.flayers[st])(RT[st][:5000].astype(np.float32))).argmax(1)
        tc = np.asarray(net.tail(st)(RT[st][:5000].astype(np.float32))).argmax(1)
        Q6 = rand_Q(RT[st].shape[1], 4, np.random.default_rng(6)); h0 = RT[st][:5000]
        hraw = h0 + ((RT[st][::-1][:5000] - h0) @ Q6) @ Q6.T; hi = proj(net.rng_[st], h0, hraw); lo_r, hi_r = net.rng_[st]
        R.setdefault('range', {})[st] = dict(unedited_outside=float(((h0 < lo_r) | (h0 > hi_r)).mean()), interchanged_clipped=float((hraw != hi).mean()),
                                             hi_bound_median=float(np.median(hi_r)), lo_bound_min=float(np.min(lo_r)))
        log(f'range {st}: {R["range"][st]}')
        ce = net.exact(st, hi)[0].argmax(1)
        R['float_tail'][st] = dict(float_unedited=float((fc == cT[:5000]).mean()), train_tail_unedited=float((tc == cT[:5000]).mean()),
                                   float_interchanged=float((np.asarray(make_ftail(net.flayers[st])(hi.astype(np.float32))).argmax(1) == ce).mean()),
                                   train_tail_interchanged=float((np.asarray(net.tail(st)(hi.astype(np.float32))).argmax(1) == ce).mean()))
        log(f'tails {st}: call agreement with exact {R["float_tail"][st]}')
    save()

    nrand = 50 if sm else (1000 if a.net in ('trained', 'pfn') else 200); steps = 100 if sm else 800
    R['stages'] = {}
    for st in net.stages:
        ft = net.tail(st); d = RT[st].shape[1]
        hb_tr, hs_tr = RF[st][fb], RF[st][fs]
        H = {arm: (RT[st][pairs[arm][2]], RT[st][pairs[arm][3]]) for arm in ('rand', 'mani')}
        res = dict(d=d, k={})
        for k in KS:
            rk = dict(das={}, perm={}, random={})
            # random null
            rr = np.random.default_rng(100 + k); null = {arm: {V: [] for V in VARS} for arm in H}; nullL = {arm: {V: [] for V in VARS} for arm in H}
            for _ in range(nrand):
                Q = rand_Q(d, k, rr)
                for arm in H:
                    c = eval_sub(net, st, *H[arm], Q, None, None)
                    for V in VARS:
                        s_ = scores(c, lab[arm][V], lab[arm]['base']); null[arm][V].append(s_['eff'] if s_['eff'] is not None else np.nan)
                        nullL[arm][V].append(s_['effL'] if s_['effL'] is not None else np.nan)
            for V in VARS:
                # DAS
                Q, info = das_fit(ft, hb_tr, hs_tr, lab['train'][V], lab['train']['base'], k, seed=10 + k, steps=steps, rg=net.rng_[st])
                out = dict(train=info)
                for arm in H:
                    c = eval_sub(net, st, *H[arm], Q, None, None); s_ = scores(c, lab[arm][V], lab[arm]['base'])
                    s_['eff_ci'] = boot_eff(c, lab[arm][V], lab[arm]['base'], B=100 if sm else 1000)
                    nv = np.asarray(null[arm][V]); fin = np.isfinite(nv)
                    s_['p_rot'] = float((1 + (nv[fin] >= s_['eff']).sum()) / (1 + fin.sum())) if s_['eff'] is not None and fin.any() else None
                    s_['transfer'] = {V2: scores(c, lab[arm][V2], lab[arm]['base'])['eff'] for V2 in VARS}
                    out[arm] = s_; out[f'_c_{arm}'] = c
                rk['das'][V] = out
                # permuted-label DAS (labels of a deranged pair)
                pr = np.random.default_rng(20 + k); pi = pr.permutation(npair)
                while (pi == np.arange(npair)).any(): pi = pr.permutation(npair)
                Qp, infop = das_fit(ft, hb_tr, hs_tr, lab['train'][V][pi], lab['train']['base'], k, seed=30 + k, steps=steps, rg=net.rng_[st])
                outp = dict(train=infop)
                for arm in H:
                    c = eval_sub(net, st, *H[arm], Qp, None, None); s_ = scores(c, lab[arm][V], lab[arm]['base'])
                    s_['eff_ci'] = boot_eff(c, lab[arm][V], lab[arm]['base'], B=100 if sm else 1000)
                    s_['das_minus_perm'] = boot_diff(out[f'_c_{arm}'], c, lab[arm][V], lab[arm]['base'], B=100 if sm else 1000)
                    outp[arm] = s_
                rk['perm'][V] = outp
                for arm in H: out.pop(f'_c_{arm}')
                for arm in H:
                    nv = np.asarray(null[arm][V], np.float64)
                    rk['random'].setdefault(arm, {})[V] = dict(mean=float(np.nanmean(nv)), q=np.nanquantile(nv, [.025, .5, .975]).tolist() if np.isfinite(nv).any() else None,
                                                             n=int(np.isfinite(nv).sum()), effL_mean=float(np.nanmean(nullL[arm][V])))
                log(f'{st} k={k} {V}: DAS eff {out["rand"]["eff"]} {out["rand"]["eff_ci"]} (n {out["rand"]["n_eff"]}), p_rot {out["rand"]["p_rot"]}; '
                    f'random {rk["random"]["rand"][V]["mean"]:.3f} {rk["random"]["rand"][V]["q"]}; perm-label {outp["rand"]["eff"]}; '
                    f'manifold DAS {out["mani"]["eff"]} (n {out["mani"]["n_eff"]}) p_rot {out["mani"]["p_rot"]}; transfer {out["rand"]["transfer"]}')
                if a.net == 'trained' and V in EDITV and k in (1, 2):
                    rk['das'][V]['_Q'] = Q
            res['k'][k] = rk; R['stages'][st] = res; save()
        # calibration (trained network only)
        if a.net == 'trained':
            res['calibration'] = calibrate(net, st, res, xF, xT, yT, RF, sm); save()
            for k in (1, 2):
                for V in EDITV: res['k'][k]['das'][V].pop('_Q', None)
        R['stages'][st] = res; save()
    log(f'wrote {JS}; X10 {a.net} DONE')


def calibrate(net, st, res, xF, xT, yT, RF, sm):
    rng = np.random.default_rng(4); n = 600 if sm else 5000; rows = rng.choice(len(xT), n, replace=False)
    xc = xT[rows]; yc = yT[rows]; isW = yc == W_
    logm = np.log(np.maximum(mass(xF), 1e-3)); pf = np.random.default_rng(0).permutation(len(xF)); nf = 2 * len(xF) // 3
    rid = RidgeR(RF['z'], logm, pf[:nf], pf[nf:])
    th = np.random.default_rng(67).uniform(0, 2 * np.pi, n); xtr = translate(xc, 0.06, th)
    m0c, m2c = moments(xc); _, m2t = moments(xtr); pred = float(np.mean(0.5 * np.log(m2t / m2c)))
    xsc = scale(xc, np.ones(n), np.full(n, np.exp(pred)))
    Rc, Rt, Rs = net.reps(xc), net.reps(xtr), net.reps(xsc)
    mc = rid(Rc['z']); cc = Rc['logits'].argmax(1)
    full = dict(pred_half_dlogM2=pred, dlogm_true_translation=float(np.mean(np.log(mass(xtr) / np.maximum(mass(xc), 1e-9)))),
                dlogmhat_translation=float(np.mean(rid(Rt['z']) - mc)), dlogmhat_scaling=float(np.mean(rid(Rs['z']) - mc)),
                WZ_base=float((cc[isW] == Z_).mean()), WZ_translation=float((Rt['logits'].argmax(1)[isW] == Z_).mean()),
                WZ_scaling=float((Rs['logits'].argmax(1)[isW] == Z_).mean()), N=n, N_W=int(isW.sum()), readout_alpha=rid.alpha)
    full['calibrated_ratio_translation'] = full['dlogmhat_translation'] / full['dlogmhat_scaling']
    log(f'calibration {st}: {full}')
    out = dict(full=full, patch={})
    d = Rc[st].shape[1]

    def patch(Q, Re):
        lo, z = net.exact(st, proj(net.rng_[st], Rc[st], Rc[st] + ((Re[st] - Rc[st]) @ Q) @ Q.T)); return rid(z) - mc, lo.argmax(1)
    rr = np.random.default_rng(55); B = 100 if sm else 1000
    for k in (1, 2):
        nullT, nullS, nullW = [], [], []
        for _ in range(B):
            Q = rand_Q(d, k, rr); dt, ct = patch(Q, Rt); ds, _ = patch(Q, Rs)
            nullT.append(dt.mean() / full['dlogmhat_translation']); nullS.append(ds.mean() / full['dlogmhat_scaling']); nullW.append((ct[isW] == Z_).mean())
        nullT, nullS, nullW = map(np.asarray, (nullT, nullS, nullW))
        for V in EDITV:
            Q = res['k'][k]['das'][V].get('_Q')
            if Q is None: continue
            dt, ct = patch(Q, Rt); ds, _ = patch(Q, Rs); fT = dt.mean() / full['dlogmhat_translation']; fS = ds.mean() / full['dlogmhat_scaling']
            wz = float((ct[isW] == Z_).mean()); cal = dt.mean() / full['dlogmhat_scaling']
            br = np.random.default_rng(9); bt = []
            dtf, dsf = rid(Rt['z']) - mc, rid(Rs['z']) - mc
            for _ in range(B):
                i = br.integers(0, n, n); bt.append((dt[i].mean() / dtf[i].mean(), ds[i].mean() / dsf[i].mean(), dt[i].mean() / dsf[i].mean()))
            bt = np.asarray(bt)
            out['patch'][f'{V}_k{k}'] = dict(frac_translation=float(fT), frac_translation_ci=np.quantile(bt[:, 0], [.025, .975]).tolist(),
                                             frac_scaling=float(fS), frac_scaling_ci=np.quantile(bt[:, 1], [.025, .975]).tolist(),
                                             calibrated_ratio=float(cal), calibrated_ratio_ci=np.quantile(bt[:, 2], [.025, .975]).tolist(),
                                             WZ_patched=wz, null_frac_translation=np.quantile(nullT, [.025, .5, .975]).tolist(),
                                             null_frac_scaling=np.quantile(nullS, [.025, .5, .975]).tolist(), null_WZ=np.quantile(nullW, [.025, .5, .975]).tolist(),
                                             p_rot_translation=float((1 + (nullT >= fT).sum()) / (1 + B)), p_rot_WZ=float((1 + (nullW >= wz).sum()) / (1 + B)))
            log(f'calibration {st} {V} k={k}: {out["patch"][f"{V}_k{k}"]}')
    return out


def summary():
    T = json.loads((OUT / 'x10_trained.json').read_text()); rows = []
    for st in STAGES:
        for k in KS:
            for V in VARS:
                e = T['stages'][st]['k'][str(k)]['das'][V]['rand']; rows.append((st, k, V, e['eff'], e['p_rot']))
    ps = np.array([r[4] if r[4] is not None else 1.0 for r in rows]); adj = holm(ps)
    S = dict(tests=[dict(stage=r[0], k=r[1], V=r[2], eff=r[3], p_rot=r[4], holm=float(h)) for r, h in zip(rows, adj)], n_tests=len(rows))
    S['n_holm_005'] = int((adj < 0.05).sum())
    for nm in ('untrained0', 'untrained1', 'untrained2', 'pfn'):
        p = OUT / f'x10_{nm}.json'
        if not p.exists(): continue
        U = json.loads(p.read_text()); S[nm] = dict(acc=U.get('acc_T'), calls=U.get('call_dist_T'))
        for st, res in U.get('stages', {}).items():
            for k in KS:
                for V in VARS:
                    e = res['k'][str(k)]['das'][V]['rand']; rn = res['k'][str(k)]['random']['rand'][V]
                    S[nm][f'{st}_k{k}_{V}'] = dict(eff=e['eff'], n_eff=e['n_eff'], p_rot=e['p_rot'], random_mean=rn['mean'], random_q=rn['q'])
    (OUT / 'x10_summary.json').write_text(json.dumps(clean(S), indent=1)); log(f'Holm: {S["n_holm_005"]} / {len(rows)} below 0.05')
    for t in S['tests']: log(t)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--net', default='trained'); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--summary', action='store_true'); a = ap.parse_args()
    summary() if a.summary else main(a)
