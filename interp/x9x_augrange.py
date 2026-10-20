"""X9x (claims A10r / A10m; open question of X9wv): what do shift-augmented set networks compute?  Augmentation-range scan
and centroid-offset read-out in PFN, EFN and Deep Sets.

Networks (stored-axis inputs, no recentring; official checkpoints untouched):
  A = 0      X9 / small-model checkpoints          pfn_64-64-32_s<k>, efn_100-100-128_s<k>, dsets-d64-n64/seed<k>
  A = 0.08   X9w -aug checkpoints                  ..._s<k>-aug, dsets-d64-n64-aug/seed<k>
  A = 0.04, 0.16, 0.30 (X9x, new)                  tags -aug04, -aug16, -aug30 (interp/x9_train.py --aug A, CPU Jobs through
             train/x9w_cpu_batch.sh, same epochs / patience / wall limit as X9w; Deep Sets train/train_small.py --aug=A, GPU Job
             train/make_jobs.py --set x9x).  Seeds 0, 1.  Augmentation: per training jet a rigid translation of all real
             particles by |c| ~ U(0, A) rad in a uniform direction, redrawn every epoch.
  untrained  2 redraws per architecture (rng 3803 + k): PFN / EFN Dense kernels he_uniform (energyflow default), biases 0;
             Deep Sets torch Linear default U(+-1/sqrt(fan_in)) for kernels and biases; Deep Sets keep the trained shift / scale.
Samples: fit / edit sample E = interp.jedi_core.sample(per_class=2000, seed=3801) (N 10,000, 2,000 true W); hold-out
  H = sample(2000, 3802) minus rows shared with E.
Pooled latent: PFN / EFN L (energyflow latent, the sum over particles of Phi), Deep Sets S = sum over real particles of phi.
(1) Edit scan on E.  T in {0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24, 0.30, 0.40} rad; translation of every real particle by T
  at 8 azimuths (0, 45, ..., 315 deg; per-jet response = mean over the 8).  Null: rotation about the stored axis with mean
  displacement T (X6 rotate_matched, rng 3810 + i; saturates at a half turn for T > 2 * mean r).  Matched mass-variable edit
  dilM2(T): angles scaled per jet by s = sqrt(1 + T^2 M0 / M2) = the azimuth-mean Delta log M2 of the translation, centroid
  direction unchanged.  Metrics per (network, T): W -> Z flow = mean over true-W jets of [Z call after edit] - [Z call before];
  % calls changed (all jets); excess d(T) = flow(T) - flow(rot_T) with 1,000-draw jet bootstrap (CI, sd); calibrated
  R(T) = flow(T) / flow(dilM2_T) and R_rot (paired bootstrap); exact McNemar on the paired W -> Z indicators, T vs rot, at
  azimuth 0 (Holm over the network x T grid per architecture).
  Break offset T* (pre-registered): first T with d(T) > 2 sd (censored = 0.5 if none).  Added after the smoke run (W -> Z flow
  is not monotone in T: W jets pass Z towards t at large offsets, and flow(dilM2) <= 0 there, so R is undefined): the same on
  calls changed, dc(T) = calls(T) - calls(rot_T) (per-jet paired, all jets), T*_calls = first T with dc > 2 sd; calibrated
  Rc(T) = dc(T) / calls(dilM2_T); T*_25 = first linear-interpolated T where Rc reaches 0.25.
  Seed-level test per architecture: Pearson r(T*, A) over the 10 trained networks; H0 T* independent of A (permutation of the
  A labels over networks, 50,000 draws); H1 T* increases with A (one-sided); pooled over architectures stratified (mean r).
  OLS slope of T* on A reported (H1 prediction: slope near 1, intercept near 0).
(2) Latent read-out.  Offset c = pT-weighted centroid (sum pT (deta, dphi) / sum pT, real particles) of the (edited) jet.
  probe_tr: ridge (standardised latent, alpha on a 20 % split of E) from the latent to c on E jets translated by a random
  vector (|t| ~ U(0, 0.4), uniform direction, rng 3820), tested on H translated the same way (rng 3821): R^2 and relative
  error per |c| bin.  probe_nat: ridge to c on unedited E jets (natural offsets), R^2 on unedited H.  Calibration: slope of
  probe_tr(c_eta) on the true c_eta among unedited H jets (natural offsets) vs its slope on +deta translations of H
  (grid t below); ratio 1 = the latent carries a translation as it carries a natural offset of the same size.
  1-dim DAS: grid t in {0, +-0.04, +-0.08, +-0.12, +-0.16, +-0.24, +-0.30, +-0.40} rad along deta (signed, so that a signed
  offset direction (t) and an even one (t^2, log M2) can be told apart); pairs (base b at t_b, source s at t_s,
  b != s, t_b != t_s), 4,000 on E for fitting, 4,000 on H for testing (rng 3830 / 3831).  Intervention on the pooled latent
  L' = L_b + u u.(L_s - L_b), |u| = 1 (raw latent coordinates); target = the network's call on b translated by t_s.
  u is fitted (jax, Adam, 3 restarts x 400 steps) to minimise the cross-entropy of the intervened output to the target
  softmax.  Effective interchange accuracy IIA = fraction of test pairs with target call != base call where the intervened
  call equals the target call (95 % pair bootstrap).  Baselines: 200 random unit u (one-sided p = (1 + #random >= DAS) / 201),
  the probe_tr c_eta direction, the same DAS on the untrained redraws.  Split: pairs with |t_b|, |t_s| <= A ('inside') and others.
  What the DAS direction carries: R^2 of u.L on H grid jets against t, t^2, log M2 and log m (invariant mass) alone.
Competing hypotheses: H_recentre (the network estimates the centroid and subtracts it): c linearly recoverable over the
  trained range, DAS IIA well above random, flow flat until the estimate saturates.  H_tolerant (moment about the stored axis
  with a weight flat over the trained range): no signed-offset subspace beyond the untrained level; DAS no better than an
  M2-like direction; excess flow growing like (T - A)^k beyond A (k fitted, descriptive).
Modes:
  AI_PY=conda ./job.sh x9x-dsx 'python -m interp.x9x_augrange --export_dsets'     (torch: new Deep Sets state_dicts -> npz)
  ./job.sh x9x-pfn-a 'python -m interp.x9x_augrange --arch pfn --part a'          -> $AI_RESULTS/jedi/x9x_pfn_a.json (parts a, b, c;
                                                                                     efn, dsets; PARTS below; Holm per part file)
  ./job.sh x9x-run-off 'python -m interp.x9x_augrange --arch official'            (official JEDI-linear accuracy + T scan, az 0)
  ./job.sh x9x-comb 'python -m interp.x9x_augrange --combine'                     -> $AI_RESULTS/jedi/x9x.json
"""
import argparse, gc, json, os, sys, time
from math import comb
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'; DSX = OUT / 'x9w_models'
SMALL = Path(os.environ.get('AI_SMALL_MODELS', '/j-jepa-vol/aaron-interp/models/small'))
W_, Z_ = 2, 3; T0 = time.time()
SEED_E, SEED_H = 3801, 3802
TS = [0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24, 0.30, 0.40]
AZ = np.deg2rad(np.arange(8) * 45.0)
TG = [0.0, 0.04, 0.08, 0.12, 0.16, 0.24, 0.30, 0.40, -0.04, -0.08, -0.12, -0.16, -0.24, -0.30, -0.40]   # signed, along deta
AUGS = {0.0: '', 0.04: '-aug04', 0.08: '-aug', 0.16: '-aug16', 0.30: '-aug30'}
SEEDS = (0, 1); NB = 1000


def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.ndarray): return clean(o.tolist())
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    return o


def mcnemar(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool); n10 = int((a & ~b).sum()); n01 = int((~a & b).sum()); n = n10 + n01
    if n == 0: return dict(a_only=0, b_only=0, p=1.0)
    k = min(n10, n01)
    from scipy.stats import binom
    return dict(a_only=n10, b_only=n01, p=float(min(1.0, 2 * binom.cdf(k, n, 0.5))))


def holm(ps):
    order = np.argsort(ps); m = len(ps); out = np.zeros(m); run = 0.0
    for r, i in enumerate(order): run = max(run, min(1.0, (m - r) * ps[i])); out[i] = run
    return out.tolist()


# ------------------------------------------------------------------------------------------------ networks
class SetNet:
    """phi (per particle, ReLU) -> weighted sum over particles (latent) -> tail MLP (ReLU, last linear).  Kernels (in, out)."""

    def __init__(self, kind, phi, tail, shift=None, scale=None, name=''):
        self.kind, self.name = kind, name
        self.phi = [(np.asarray(W, np.float32), np.asarray(b, np.float32)) for W, b in phi]
        self.tail = [(np.asarray(W, np.float64), np.asarray(b, np.float64)) for W, b in tail]
        self.shift = None if shift is None else np.asarray(shift, np.float32); self.scale = None if scale is None else np.asarray(scale, np.float32)
        self.ell = self.phi[-1][0].shape[1]

    def prep(self, x):
        x = np.asarray(x, np.float32); real = x[..., 0] > 0
        if self.kind == 'efn':
            pt = x[..., 0] * real; z = pt / np.maximum(pt.sum(1, keepdims=True), 1e-12); return z, x[..., 1:3] * real[..., None]
        a = ((x - self.shift) / self.scale) * real[..., None]
        if self.kind == 'pfn': return np.any(a != 0, -1).astype(np.float32), a     # energyflow masks rows equal to 0
        return real.astype(np.float32), a

    def latent(self, x, batch=2000):
        """phi on the real particles only (padding rows carry weight 0 in all three pools), summed per jet"""
        out = []
        for i in range(0, len(x), batch):
            xb = np.asarray(x[i:i + batch], np.float32); w, a = self.prep(xb); real = xb[..., 0] > 0
            assert real.any(1).all(), 'jet without real particles'
            h = a[real]
            for W, b in self.phi: h = np.maximum(h @ W + b, 0)
            starts = np.concatenate([[0], np.cumsum(real.sum(1))[:-1]])
            out.append(np.add.reduceat((h * w[real][:, None]).astype(np.float64), starts, axis=0))
        return np.concatenate(out)

    def head(self, L):
        h = np.asarray(L, np.float64)
        for k, (W, b) in enumerate(self.tail):
            h = h @ W + b
            if k < len(self.tail) - 1: h = np.maximum(h, 0)
        return h

    def logits(self, x): return self.head(self.latent(x))


def flow_name(arch, seed, tag): return f'{"pfn_64-64-32" if arch == "pfn" else "efn_100-100-128"}_s{seed}{tag}'


def load_net(arch, A, seed):
    tag = AUGS[A]
    if arch in ('pfn', 'efn'):
        from interp.x9_efn_pfn import Flow
        F = Flow(flow_name(arch, seed, tag)); inf = F.info
        assert not inf.get('recentre') and abs(float(inf.get('aug', 0)) - A) < 1e-9, inf
        net = SetNet(arch, F.phi, F.F + [F.out], F.shift, F.scale, name=F.name)
        return net, dict(dir=str(F.dir), acc_test=inf['acc_test'], epochs_run=inf['epochs_run'], best_epoch=inf['best_epoch'], aug_flag=inf.get('aug')), F
    d = SMALL / f'dsets-d64-n64{tag}' / f'seed{seed}'; t = json.loads((d / 'test.json').read_text())
    ar = json.loads((d / 'arch.json').read_text())['inputs']
    assert not ar.get('recentre') and abs(float(ar.get('aug', 0) or 0) - A) < 1e-9, ar
    P = np.load(DSX / f'dsets-d64-n64{tag}_s{seed}.npz')
    net = SetNet('dsets', [(P['phi.0.weight'].T, P['phi.0.bias']), (P['phi.2.weight'].T, P['phi.2.bias'])],
                 [(P[f'{k}.weight'].T, P[f'{k}.bias']) for k in ('rho.0', 'rho.2', 'rho.4', 'head')], P['_shift'], P['_scale'], name=d.name)
    return net, dict(dir=str(d), acc_test=t['accuracy'], best_epoch=t.get('best_epoch'), aug_flag=ar.get('aug')), P


def untrained(arch, k, like):
    r = np.random.default_rng(3803 + k)
    if arch in ('pfn', 'efn'):
        hu = lambda W: r.uniform(-1, 1, W.shape) * np.sqrt(6.0 / W.shape[0])
        phi = [(hu(W), np.zeros_like(b)) for W, b in like.phi]; tail = [(hu(W), np.zeros_like(b)) for W, b in like.tail]
    else:
        tu = lambda W, b: (r.uniform(-1, 1, W.shape) / np.sqrt(W.shape[0]), r.uniform(-1, 1, b.shape) / np.sqrt(W.shape[0]))
        phi = [tu(W, b) for W, b in like.phi]; tail = [tu(W, b) for W, b in like.tail]
    return SetNet(arch, phi, tail, like.shift, like.scale, name=f'{arch}_untrained_r{k}')


# ------------------------------------------------------------------------------------------------ physics / edits
def centroid(x):
    x = np.asarray(x, np.float64); pt = np.where(x[..., 0] > 0, x[..., 0], 0)
    return (pt[..., None] * x[..., 1:3]).sum(1) / np.maximum(pt.sum(1), 1e-12)[:, None]


def samples():
    from interp.jedi_core import sample
    from interp.x9w_repr_2x2 import sample_rows
    xe, ye, _ = sample(2000, SEED_E); xh, yh, _ = sample(2000, SEED_H)
    ie, ih = sample_rows(2000, SEED_E), sample_rows(2000, SEED_H); keep = ~np.isin(ih, ie)
    return np.ascontiguousarray(xe, np.float32), ye, np.ascontiguousarray(xh[keep], np.float32), yh[keep], int((~keep).sum())


# ------------------------------------------------------------------------------------------------ (1) edit scan
def boot_idx(n, B=NB, seed=3840): return np.random.default_rng(seed).integers(0, n, (B, n))


def ci(v, I):
    m = v[I].mean(1); return [float(np.quantile(m, .025)), float(np.quantile(m, .975))], float(m.std())


def ratio_ci(a, b, I):
    ma, mb = a[I].mean(1), b[I].mean(1); ok = mb > 0; r = ma[ok] / mb[ok]
    return [float(np.quantile(r, .025)), float(np.quantile(r, .975))] if ok.mean() > 0.9 else [None, None]


def edit_scan(net, x, y, ROT, DIL):
    from interp.x9_efn_pfn import translate
    W = y == W_; I = boot_idx(int(W.sum()))
    lo0 = net.logits(x); c0 = lo0.argmax(1); z0 = (c0[W] == Z_).astype(float)
    o = dict(acc=float((c0 == y).mean()), W_to_Z_unedited=float(z0.mean()), per_T={})
    for i, T in enumerate(TS):
        zs = np.zeros(int(W.sum())); chs = np.zeros(len(y)); z_az0 = None; Ia = boot_idx(len(y), seed=3842)
        for th in AZ:
            ce = net.logits(translate(x, T, th)).argmax(1); ze = (ce[W] == Z_).astype(float); zs += ze; chs += ce != c0
            if z_az0 is None: z_az0 = ze.astype(bool)
        fT = zs / len(AZ) - z0; chT = chs / len(AZ)
        cr = net.logits(ROT[i]).argmax(1); zr = cr[W] == Z_; fR = zr - z0
        cd = net.logits(DIL[i]).argmax(1); zd = cd[W] == Z_; fD = zd - z0
        d = fT - fR; dci, dsd = ci(d, I); chR = (cr != c0).astype(float); chD = (cd != c0).astype(float); dc = chT - chR; dcci, dcsd = ci(dc, Ia)
        e = dict(flow_T=float(fT.mean()), flow_T_ci=ci(fT, I)[0], flow_rot=float(fR.mean()), flow_rot_ci=ci(fR, I)[0],
                 flow_dil=float(fD.mean()), flow_dil_ci=ci(fD, I)[0], excess=float(d.mean()), excess_ci=dci, excess_sd=dsd,
                 calls_T=float(chT.mean()), calls_rot=float((cr != c0).mean()), calls_dil=float((cd != c0).mean()),
                 W_to_Z_T=float((zs / len(AZ)).mean()),
                 R=float(fT.mean() / fD.mean()) if fD.mean() > 0 else None, R_ci=ratio_ci(fT, fD, I) if fD.mean() > 0 else [None, None],
                 R_rot=float(fR.mean() / fD.mean()) if fD.mean() > 0 else None,
                 Rex=float(d.mean() / fD.mean()) if fD.mean() > 0 else None, Rex_ci=ratio_ci(d, fD, I) if fD.mean() > 0 else [None, None],
                 excess_calls=float(dc.mean()), excess_calls_ci=dcci, excess_calls_sd=dcsd, calls_T_ci=ci(chT, Ia)[0], calls_rot_ci=ci(chR, Ia)[0],
                 calls_dil_ci=ci(chD, Ia)[0], Rc=float(dc.mean() / chD.mean()) if chD.mean() > 0 else None,
                 Rc_ci=ratio_ci(dc, chD, Ia) if chD.mean() > 0 else [None, None],
                 mcnemar_T_vs_rot_az0=mcnemar(z_az0, zr))
        o['per_T'][str(T)] = e
    return o


def break_offsets(o):
    ex = [o['per_T'][str(T)] for T in TS]
    t2 = next((T for T, e in zip(TS, ex) if e['excess'] > 2 * e['excess_sd'] and e['excess_sd'] > 0), None)
    tc = next((T for T, e in zip(TS, ex) if e['excess_calls'] > 2 * e['excess_calls_sd'] and e['excess_calls_sd'] > 0), None)
    rr = [(e['Rc'] if e['Rc'] is not None else 0.0) for e in ex]; t25 = None
    for k, (T, r) in enumerate(zip(TS, rr)):
        if r >= 0.25:
            if k == 0: t25 = T
            else: t25 = TS[k - 1] + (0.25 - rr[k - 1]) / (r - rr[k - 1]) * (T - TS[k - 1])
            break
    return dict(T_star_2sd=t2 if t2 is not None else 0.5, T_star_2sd_censored=t2 is None,
                T_star_calls=tc if tc is not None else 0.5, T_star_calls_censored=tc is None,
                T_star_25=t25 if t25 is not None else 0.5, T_star_25_censored=t25 is None)


def growth_exponent(o, A):
    pts = [(T - A, o['per_T'][str(T)]['excess']) for T in TS if T > A + 1e-9 and o['per_T'][str(T)]['excess'] > 0.01]
    if len(pts) < 3: return None
    u, v = np.log([p[0] for p in pts]), np.log([p[1] for p in pts]); k = np.polyfit(u, v, 1)[0]; return float(k)


# ------------------------------------------------------------------------------------------------ (2) latent read-out
def ridge_fit(X, Y, seed=3850):
    mu, sd = X.mean(0), X.std(0) + 1e-9; Xs = (X - mu) / sd; ym = Y.mean(0)
    r = np.random.default_rng(seed); p = r.permutation(len(X)); nv = len(X) // 5; iv, it = p[:nv], p[nv:]
    U, s, Vt = np.linalg.svd(Xs[it], full_matrices=False); best = None
    for a in 10.0 ** np.arange(-4, 4.1, 0.5):
        B = Vt.T @ np.diag(s / (s ** 2 + a)) @ U.T @ (Y[it] - ym); e = ((Xs[iv] @ B + ym - Y[iv]) ** 2).sum()
        if best is None or e < best[0]: best = (e, a)
    a = best[1]; U, s, Vt = np.linalg.svd(Xs, full_matrices=False); B = Vt.T @ np.diag(s / (s ** 2 + a)) @ U.T @ (Y - ym)
    return dict(mu=mu, sd=sd, B=B, ym=ym, alpha=float(a))


def ridge_pred(P, X): return (X - P['mu']) / P['sd'] @ P['B'] + P['ym']


def r2(p, t): return float(1 - ((p - t) ** 2).sum() / ((t - t.mean(0)) ** 2).sum())


def tail_jax(net):
    import jax.numpy as jnp
    Ws = [(jnp.asarray(W, jnp.float32), jnp.asarray(b, jnp.float32)) for W, b in net.tail]
    def f(L):
        h = L
        for k, (W, b) in enumerate(Ws):
            h = h @ W + b
            if k < len(Ws) - 1: h = jnp.maximum(h, 0)
        return h
    return f


def das_fit(net, Lb, Ls, tgt, steps=400, restarts=3, seed=3860):
    import jax, jax.numpy as jnp
    f = tail_jax(net); Lb, Ls = jnp.asarray(Lb, jnp.float32), jnp.asarray(Ls, jnp.float32)
    pt = jax.nn.softmax(jnp.asarray(tgt, jnp.float32), -1)
    def loss(v):
        u = v / jnp.linalg.norm(v); Lp = Lb + (u[None, :] * ((Ls - Lb) @ u)[:, None]); lo = f(Lp)
        return -(pt * jax.nn.log_softmax(lo, -1)).sum(-1).mean()
    g = jax.jit(jax.value_and_grad(loss)); best = None; r = np.random.default_rng(seed)
    for k in range(restarts):
        v = jnp.asarray(r.standard_normal(Lb.shape[1]), jnp.float32); m = jnp.zeros_like(v); s2 = jnp.zeros_like(v)
        for t in range(1, steps + 1):
            l, gr = g(v); m = 0.9 * m + 0.1 * gr; s2 = 0.999 * s2 + 0.001 * gr ** 2
            v = v - 0.05 * (m / (1 - 0.9 ** t)) / (jnp.sqrt(s2 / (1 - 0.999 ** t)) + 1e-8); v = v / jnp.linalg.norm(v)
        l = float(loss(v))
        if best is None or l < best[0]: best = (l, np.asarray(v, np.float64))
    return best[1] / np.linalg.norm(best[1]), best[0]


def iia(net, u, Lb, Ls, cb, ct):
    Lp = Lb + u[None, :] * ((Ls - Lb) @ u)[:, None]; ci_ = net.head(Lp).argmax(1); eff = ct != cb
    return ci_ == ct, eff


def latent_readout(net, xe, xh, A, A_known=True):
    from interp.x9_efn_pfn import translate, moments
    o = {}
    # probes
    re_, rh_ = np.random.default_rng(3820), np.random.default_rng(3821)
    te = (re_.uniform(0, 0.4, len(xe)), re_.uniform(0, 2 * np.pi, len(xe))); th_ = (rh_.uniform(0, 0.4, len(xh)), rh_.uniform(0, 2 * np.pi, len(xh)))
    def trans_vec(x, r, th):
        e = x.copy(); rl = e[..., 0] > 0
        e[..., 1] += (r * np.cos(th))[:, None].astype(np.float32) * rl; e[..., 2] += (r * np.sin(th))[:, None].astype(np.float32) * rl; return e
    xte, xth = trans_vec(xe, *te), trans_vec(xh, *th_)
    Lte, Lth = net.latent(xte), net.latent(xth); cte, cth = centroid(xte), centroid(xth)
    P = ridge_fit(Lte, cte); ph = ridge_pred(P, Lth); mag = np.hypot(*cth.T)
    bins = [0, 0.04, 0.08, 0.16, 0.30, 0.42]; rel = {}
    for a, b in zip(bins[:-1], bins[1:]):
        k = (mag >= a) & (mag < b); rel[f'{a}-{b}'] = dict(n=int(k.sum()), rel_err=float(np.sqrt(((ph[k] - cth[k]) ** 2).sum(1).mean()) / np.sqrt((cth[k] ** 2).sum(1).mean())))
    o['probe_tr'] = dict(alpha=P['alpha'], R2_holdout=r2(ph, cth), rel_err_by_offset=rel)
    L0e, L0h = net.latent(xe), net.latent(xh); c0e, c0h = centroid(xe), centroid(xh)
    Pn = ridge_fit(L0e, c0e); o['probe_nat'] = dict(alpha=Pn['alpha'], R2_holdout=r2(ridge_pred(Pn, L0h), c0h),
                                                     natural_offset_sd=[float(v) for v in c0h.std(0)],
                                                     natural_offset_median_abs=float(np.median(np.hypot(*c0h.T))))
    # calibration of probe_tr: natural offsets vs +deta translations of H
    pn = ridge_pred(P, L0h)[:, 0]; slope_nat = float(np.polyfit(c0h[:, 0], pn, 1)[0])
    # grid latents along +deta
    GE = [net.latent(translate(xe, t, 0.0)) for t in TG]; GH = [net.latent(translate(xh, t, 0.0)) for t in TG]
    tt = np.concatenate([np.full(len(xh), t) for t in TG]); pg = np.concatenate([ridge_pred(P, g)[:, 0] for g in GH])
    slope_tr = float(np.polyfit(tt, pg, 1)[0])
    o['calibration'] = dict(slope_on_natural_offsets=slope_nat, slope_on_translations=slope_tr,
                            ratio_translation_over_natural=slope_tr / slope_nat if abs(slope_nat) > 1e-9 else None)
    CE = [net.head(g).argmax(1) for g in GE]; LE = [net.head(g) for g in GE]; CH = [net.head(g).argmax(1) for g in GH]
    def pairs(n, seed):
        r = np.random.default_rng(seed); b = r.integers(0, n, 4000); s = (b + r.integers(1, n, 4000)) % n
        tb = r.integers(0, len(TG), 4000); ts = (tb + r.integers(1, len(TG), 4000)) % len(TG); return b, s, tb, ts
    b, s, tb, ts = pairs(len(xe), 3830)
    Lb = np.stack([GE[i][j] for i, j in zip(tb, b)]); Ls = np.stack([GE[i][j] for i, j in zip(ts, s)]); tg = np.stack([LE[i][j] for i, j in zip(ts, b)])
    u, lfit = das_fit(net, Lb, Ls, tg)
    b, s, tb, ts = pairs(len(xh), 3831)
    Lb = np.stack([GH[i][j] for i, j in zip(tb, b)]); Ls = np.stack([GH[i][j] for i, j in zip(ts, s)])
    cb = np.array([CH[i][j] for i, j in zip(tb, b)]); ct = np.array([CH[i][j] for i, j in zip(ts, b)])
    ok, eff = iia(net, u, Lb, Ls, cb, ct); inside = (np.abs(np.array(TG)[tb]) <= A + 1e-9) & (np.abs(np.array(TG)[ts]) <= A + 1e-9)
    I = np.random.default_rng(3841).integers(0, int(eff.sum()), (NB, int(eff.sum()))) if eff.sum() > 0 else None
    ve = ok[eff].astype(float)
    das = dict(train_loss=lfit, n_pairs=4000, n_effective=int(eff.sum()), iia_eff=float(ve.mean()) if eff.sum() else None,
               iia_eff_ci=ci(ve, I)[0] if I is not None else [None, None], iia_all=float(ok.mean()),
               n_eff_inside=int((eff & inside).sum()), iia_eff_inside=float(ok[eff & inside].mean()) if (eff & inside).sum() else None,
               n_eff_outside=int((eff & ~inside).sum()), iia_eff_outside=float(ok[eff & ~inside].mean()) if (eff & ~inside).sum() else None)
    rr = np.random.default_rng(3870); rnd = []
    for _ in range(200):
        v = rr.standard_normal(net.ell); v /= np.linalg.norm(v); okr, _ = iia(net, v, Lb, Ls, cb, ct); rnd.append(float(okr[eff].mean()) if eff.sum() else np.nan)
    rnd = np.asarray(rnd)
    das['random'] = dict(mean=float(np.nanmean(rnd)), q95=float(np.nanquantile(rnd, .95)), max=float(np.nanmax(rnd)),
                         p=float((1 + (rnd >= das['iia_eff'] - 1e-12).sum()) / 201) if das['iia_eff'] is not None else None)
    wp = P['B'][:, 0] / P['sd']; wp /= np.linalg.norm(wp); okp, _ = iia(net, wp, Lb, Ls, cb, ct)
    das['probe_direction_iia_eff'] = float(okp[eff].mean()) if eff.sum() else None
    # what u.L carries on the H grid
    proj = np.concatenate([g @ u for g in GH]); Ms = [moments(translate(xh, t, 0.0)) for t in TG]
    lm2 = np.concatenate([np.log(np.maximum(M['M2'], 1e-30)) for M in Ms]); lm = np.concatenate([M['logm'] for M in Ms])
    def r2_1(z):
        Z = np.stack([z, np.ones_like(z)], 1); c, *_ = np.linalg.lstsq(Z, proj, rcond=None); return r2(Z @ c, proj)
    das['carries_R2'] = dict(t=r2_1(tt), t2=r2_1(tt ** 2), logM2=r2_1(lm2), logm=r2_1(lm))
    o['das'] = das
    return o


# ------------------------------------------------------------------------------------------------ jobs
PARTS = {'a': [(0.0, 0), (0.0, 1), (0.04, 0), (0.04, 1)], 'b': [(0.08, 0), (0.08, 1), (0.16, 0), (0.16, 1)],
         'c': [(0.30, 0), (0.30, 1), ('untrained', 0), ('untrained', 1)]}


def run_arch(arch, smoke=False, part=''):
    from interp.x9_efn_pfn import rotate_matched, moments, scale_angles
    xe, ye, xh, yh, nshared = samples()
    if smoke: xe, ye, xh, yh = xe[::10], ye[::10], xh[::10], yh[::10]
    log(f'samples: E {len(ye)} (W {int((ye == W_).sum())}), H {len(yh)} ({nshared} shared rows removed)')
    M = moments(xe); ROT = [rotate_matched(xe, T, np.random.default_rng(3810 + i)) for i, T in enumerate(TS)]
    DIL = [scale_angles(xe, np.sqrt(1 + T ** 2 * M['M0'] / np.maximum(M['M2'], 1e-30))) for T in TS]
    hyg = dict(dil_scale_mean={str(T): float(np.sqrt(1 + T ** 2 * M['M0'] / np.maximum(M['M2'], 1e-30)).mean()) for T in TS},
               natural_offset_median=float(np.median(np.hypot(*centroid(xe).T))),
               rot_mean_displacement={str(T): float(np.median(np.hypot(*(ROT[i][..., 1:3] - xe[..., 1:3]).transpose(2, 0, 1))[xe[..., 0] > 0])) for i, T in enumerate(TS)})
    log('hygiene', json.dumps(clean(hyg)))
    R = dict(header='see interp/x9x_augrange.py docstring', arch=arch, TS=TS, TG=TG, AZ_deg=list(np.rad2deg(AZ)),
             sample=dict(E=dict(per_class=2000, seed=SEED_E, N=len(ye)), H=dict(per_class=2000, seed=SEED_H, N=len(yh), shared_removed=nshared)),
             hygiene=hyg, nets={})
    fo = OUT / f'x9x_{arch}{"_smoke" if smoke else ""}{"_" + part if part else ""}.json'; save = lambda: fo.write_text(json.dumps(clean(R), indent=1))
    like = None
    todo = [(A, s) for A in AUGS for s in SEEDS] + [('untrained', k) for k in (0, 1)]
    if smoke: todo = [(0.0, 0), (0.08, 0), ('untrained', 0)]
    if part: todo = PARTS[part]
    for A, s in todo:
        key = f'{arch}_A{A}_s{s}' if A != 'untrained' else f'{arch}_untrained_r{s}'
        try:
            if A == 'untrained' and like is None: like = load_net(arch, 0.0, 0)[0]
            if A == 'untrained': net, info = untrained(arch, s, like), dict(untrained=True)
            else:
                net, info, raw = load_net(arch, A, s); like = like or net
                if arch in ('pfn', 'efn'):    # numpy forward vs the TF softmax written at training time (sample(2000, 313))
                    from interp.x9_efn_pfn import softmax
                    from interp.jedi_core import sample
                    ck = np.load(Path(info['dir']) / 'check.npz'); xc, _, _ = sample(2000, 313)
                    info['check_tf_max_abs_dp'] = float(np.abs(softmax(net.logits(xc)) - ck['prob']).max())
                else:
                    cs = int(raw['_check_seed']) if '_check_seed' in raw.files else 909   # X9w exports: sample(2000, 909)
                    from interp.jedi_core import sample
                    xc = sample(2000, cs)[0]
                    info['check_torch_max_abs_dlogit'] = float(np.abs(net.logits(xc) - raw['_check_logits']).max()); info['check_seed'] = cs
        except Exception as e:  # noqa: BLE001
            log('SKIP', key, repr(e)[:300]); R['nets'][key] = dict(error=repr(e)[:500]); save(); continue
        o = edit_scan(net, xe, ye, ROT, DIL); o.update(info=info, A=A if A != 'untrained' else None, seed=s, untrained=A == 'untrained')
        if A != 'untrained': o['break'] = break_offsets(o); o['growth_exponent_beyond_A'] = growth_exponent(o, A)
        log(f'{key}: acc {o["acc"]:.4f} (test {info.get("acc_test")}); W->Z0 {o["W_to_Z_unedited"]:.3f}; excess ' +
            ' '.join(f'{T}:{o["per_T"][str(T)]["excess"]:.3f}' for T in TS) + '; excess calls ' +
            ' '.join(f'{T}:{o["per_T"][str(T)]["excess_calls"]:.3f}' for T in TS) + '; Rc ' +
            ' '.join(f'{T}:{o["per_T"][str(T)]["Rc"]:.2f}' if o["per_T"][str(T)]["Rc"] is not None else f'{T}:-' for T in TS) +
            f'; break {o.get("break")}; checks {json.dumps({k: v for k, v in info.items() if k.startswith("check")})}')
        save()
        o['readout'] = latent_readout(net, xe, xh, A if A != 'untrained' else 0.0)
        d = o['readout']['das']
        log(f'{key}: probe_tr R2 {o["readout"]["probe_tr"]["R2_holdout"]:.3f} rel_err {json.dumps({k: round(v["rel_err"], 3) for k, v in o["readout"]["probe_tr"]["rel_err_by_offset"].items()})}; '
            f'probe_nat R2 {o["readout"]["probe_nat"]["R2_holdout"]:.3f}; calib {json.dumps(clean(o["readout"]["calibration"]))}; '
            f'DAS iia_eff {d["iia_eff"]} (n {d["n_effective"]}; inside {d["iia_eff_inside"]} n {d["n_eff_inside"]}, outside {d["iia_eff_outside"]}) '
            f'random mean {d["random"]["mean"]:.3f} q95 {d["random"]["q95"]:.3f} p {d["random"]["p"]}; probe-dir {d["probe_direction_iia_eff"]}; carries {json.dumps(clean(d["carries_R2"]))}')
        R['nets'][key] = o; save(); del net; gc.collect()
    # Holm over the network x T grid (T vs rot McNemar, azimuth 0)
    keys = [(k, T) for k, v in R['nets'].items() if 'per_T' in v and not v['untrained'] for T in TS]
    h = holm([R['nets'][k]['per_T'][str(T)]['mcnemar_T_vs_rot_az0']['p'] for k, T in keys])
    for (k, T), hh in zip(keys, h): R['nets'][k]['per_T'][str(T)]['mcnemar_T_vs_rot_az0']['p_holm'] = hh
    R['holm_m'] = len(keys); save(); log('wrote', fo, 'X9x', arch, 'DONE')


def run_official():
    from interp.x12_retrained_jedi import Net
    from interp.x9_efn_pfn import translate, rotate_matched
    xe, ye, _, _, _ = samples(); net = Net('official'); W = ye == W_
    lo0 = net.run(xe)['logits']; c0 = lo0.argmax(1); z0 = (c0[W] == Z_).astype(float); k = int((c0 == ye).sum())
    from interp.x9w_repr_2x2 import wilson
    o = dict(acc=k / len(ye), acc_ci=wilson(k, len(ye)), W_to_Z_unedited=float(z0.mean()), per_T={})
    for i, T in enumerate(TS):
        ze = (net.run(translate(xe, T, 0.0))['logits'].argmax(1)[W] == Z_).astype(float)
        zr = (net.run(rotate_matched(xe, T, np.random.default_rng(3810 + i)))['logits'].argmax(1)[W] == Z_).astype(float)
        o['per_T'][str(T)] = dict(flow_T_az0=float((ze - z0).mean()), flow_rot=float((zr - z0).mean()))
        log(f'official T {T}: flow az0 {o["per_T"][str(T)]["flow_T_az0"]:.3f}, rot {o["per_T"][str(T)]["flow_rot"]:.3f}')
    log(f'official JEDI-linear acc {o["acc"]:.4f} {o["acc_ci"]}')
    (OUT / 'x9x_official.json').write_text(json.dumps(clean(o), indent=1)); log('X9x official DONE')


def export_dsets():
    import torch, importlib.util
    spec = importlib.util.spec_from_file_location('small_models', ROOT / 'train' / 'models.py'); Mm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(Mm); DSX.mkdir(parents=True, exist_ok=True)
    from interp.jedi_core import sample
    x, y, _ = sample(2000, SEED_E)
    for A, tag in AUGS.items():
        if tag in ('', '-aug'): continue
        for s in SEEDS:
            d = SMALL / f'dsets-d64-n64{tag}' / f'seed{s}'
            if not (d / 'best.pt').exists(): log('missing', d); continue
            arch = json.loads((d / 'arch.json').read_text()); net = Mm.build_model(arch['spec'], arch['n_particles'], cls_token=False, final_ln=arch['final_ln'])
            sd = torch.load(d / 'best.pt', map_location='cpu'); net.load_state_dict(sd); net.eval()
            P = {k: v.numpy() for k, v in sd.items()}; P['_shift'] = np.asarray(arch['inputs']['shift'], np.float32); P['_scale'] = np.asarray(arch['inputs']['scale'], np.float32)
            m = x[..., 0] > 0; xn = ((x - P['_shift']) / P['_scale']) * m[..., None]
            with torch.no_grad(): lo = net(torch.from_numpy(xn.astype(np.float32)), torch.from_numpy(m)).double().numpy()
            P['_check_logits'] = lo; P['_check_seed'] = np.array(SEED_E); np.savez(DSX / f'dsets-d64-n64{tag}_s{s}.npz', **P)
            log('exported', d, 'aug', arch['inputs'].get('aug'), 'acc on sample', float((lo.argmax(1) == y).mean()))


# ------------------------------------------------------------------------------------------------ combine
def perm_corr(T, A, draws=50000, rng=None):
    T, A = np.asarray(T, float), np.asarray(A, float); r0 = np.corrcoef(A, T)[0, 1] if T.std() > 0 else 0.0
    rng = rng or np.random.default_rng(3880); c = 0
    for _ in range(draws):
        Ap = rng.permutation(A); r = np.corrcoef(Ap, T)[0, 1] if T.std() > 0 else 0.0; c += r >= r0 - 1e-12
    return float(r0), float((1 + c) / (1 + draws))


def last_below(v):
    """Rc (with CI) and calls changed at the largest scanned T strictly below A (None for A <= 0.02)"""
    Tb = [T for T in TS if T < v['A'] - 1e-9]
    if not Tb: return None
    e = v['per_T'][str(Tb[-1])]; return dict(T=Tb[-1], Rc=e['Rc'], Rc_ci=e['Rc_ci'], excess_calls=e['excess_calls'], excess_calls_ci=e['excess_calls_ci'])


def growth_calls(v):
    """descriptive: slope of log Rc on log(T - A) over T > A with 0.02 < Rc < 0.8 (before saturation); None if < 2 points"""
    pts = [(T - v['A'], v['per_T'][str(T)]['Rc']) for T in TS if T > v['A'] + 1e-9 and (v['per_T'][str(T)]['Rc'] or 0) > 0.02
           and v['per_T'][str(T)]['Rc'] < 0.8]
    if len(pts) < 2: return None
    return float(np.polyfit(np.log([p[0] for p in pts]), np.log([p[1] for p in pts]), 1)[0])


def combine():
    R = dict(header='X9x combined: T* vs A per architecture and pooled (see interp/x9x_augrange.py)', per_arch={})
    pooled = {'T_star_2sd': {}, 'T_star_calls': {}, 'T_star_25': {}}
    for arch in ('pfn', 'efn', 'dsets'):
        D = {'nets': {}}
        for pt in ('', '_a', '_b', '_c'):
            f = OUT / f'x9x_{arch}{pt}.json'
            if f.exists(): D['nets'].update(json.loads(f.read_text())['nets'])
        rows = [v for v in D['nets'].values() if 'break' in v]
        if not rows: log('missing', arch); continue
        R.setdefault('untrained', {})[arch] = {k: dict(acc=v['acc'], das=v.get('readout', {}).get('das'), probe_tr_R2=v.get('readout', {}).get('probe_tr', {}).get('R2_holdout'))
                                              for k, v in D['nets'].items() if v.get('untrained')}
        ent = dict(networks=[dict(A=v['A'], seed=v['seed'], acc=v['acc'], acc_test=v['info'].get('acc_test'), **v['break'],
                                  growth_exponent=v.get('growth_exponent_beyond_A'),
                                  max_Rc_below_A=max([v['per_T'][str(T)]['Rc'] or 0 for T in TS if T < v['A'] - 1e-9], default=None),
                                  Rc_last_below_A=last_below(v), growth_exponent_calls=growth_calls(v),
                                  das=v.get('readout', {}).get('das'), probe_tr_R2=v.get('readout', {}).get('probe_tr', {}).get('R2_holdout'),
                                  probe_nat_R2=v.get('readout', {}).get('probe_nat', {}).get('R2_holdout'),
                                  calibration=v.get('readout', {}).get('calibration')) for v in rows])
        A = [v['A'] for v in rows]
        for m in ('T_star_2sd', 'T_star_calls', 'T_star_25'):
            T = [v['break'][m] for v in rows]; r, p = perm_corr(T, A)
            sl, ic = np.polyfit(A, T, 1) if np.std(T) > 0 else (0.0, float(np.mean(T)))
            ent[m] = dict(pearson_r=r, p_one_sided=p, slope=float(sl), intercept=float(ic), n=len(T)); pooled[m][arch] = (T, A)
        R['per_arch'][arch] = ent
        log(arch, json.dumps(clean({m: ent[m] for m in ('T_star_2sd', 'T_star_calls', 'T_star_25')})))
    rng = np.random.default_rng(3881)
    for m, d in pooled.items():
        if len(d) < 2: continue
        obs = np.mean([np.corrcoef(A, T)[0, 1] if np.std(T) > 0 else 0 for T, A in d.values()]); c = 0
        for _ in range(20000):
            st = np.mean([np.corrcoef(rng.permutation(A), T)[0, 1] if np.std(T) > 0 else 0 for T, A in d.values()]); c += st >= obs - 1e-12
        R.setdefault('pooled', {})[m] = dict(mean_r=float(obs), p_one_sided=float((1 + c) / 20001), archs=list(d))
        log('pooled', m, R['pooled'][m])
    (OUT / 'x9x.json').write_text(json.dumps(clean(R), indent=1)); log('X9x combine DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--arch', choices=['pfn', 'efn', 'dsets', 'official'])
    ap.add_argument('--export_dsets', action='store_true'); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--part', default='', choices=['', 'a', 'b', 'c']); ap.add_argument('--combine', action='store_true'); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.export_dsets: export_dsets()
    elif a.combine: combine()
    elif a.arch == 'official': run_official()
    else: run_arch(a.arch, a.smoke, a.part)
