"""X11 (claim A11; review weakness 4 and 2): preregistered blind atlas. Are JEDI-linear's S and h4 units individually
nameable as (pT, r) hinge objects more often than random rotations of the same activations, and comparably to SAE features,
while Deep Sets / PFN pooled units need an SAE to reach that rate?

The frozen plan is interp/x11_prereg.json (copied to $AI_RESULTS/jedi/x11_prereg.json at the first run; later runs check it is
unchanged). In short:
  models   jedi (S, h4), dsets (dsets-d64 seed 0, S), pfn (PFN 64-64-32 seed 0, L)
  jets     fit = sample(5000, 0) (2/3 fit-train, 1/3 fit-dev); test = sample(5000, 29) minus the fit jets; causal = 5,000 test jets
           whose edits stay in JEDI-linear's l1 input range
  bases    native live units; 20 raw rotations; 10 z-scored rotations; PCA; SAE 2x / 4x (top-k = d by mean f², and all alive)
  name     MARS (<= 8 terms) on the E11 hand-built rank-window library; held-out R² >= 0.9
  causal   pT x 1.1 or angles x 1.1 in the top term's rank window; OLS slope actual-on-predicted, CI inside [0.8, 1.2]
Usage: python -m interp.x11_sae_atlas --model jedi|dsets|pfn     -> $AI_RESULTS/jedi/x11_<model>.json
       python -m interp.x11_sae_atlas --combine                   -> x11_summary.json, x11_atlas.png
Options: --smoke (small sizes, writes x11_<model>_smoke.json)
"""
import argparse, json, os, re, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'; OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
PRE = json.loads((Path(__file__).resolve().parent / 'x11_prereg.json').read_text())
R2_NAMED, SLOPE_LO, SLOPE_HI, MAX_TERMS, N_SEL, N_ROT, N_ROT_STD = 0.90, 0.8, 1.2, 8, 4000, 20, 10
B, BC, N_CAUSAL, EDIT_K, MIN_PRED = 500, 500, 5000, 1.1, 0.02
LAMBDAS = [0.3, 0.1, 0.03, 0.01, 0.003]


def log(*a): print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return clean(o.tolist())
    return o


def bh(p):
    p = np.asarray(p, np.float64); m = len(p)
    if m == 0: return p
    o = np.argsort(p); adj = np.empty(m); run = 1.0
    for r in range(m - 1, -1, -1): run = min(run, p[o[r]] * m / (r + 1)); adj[o[r]] = run
    return adj


def holm(p):
    p = np.asarray(p, np.float64); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def ci(v): v = np.asarray(v, np.float64); return [float(np.quantile(v, .025)), float(np.quantile(v, .975))]


def boot_p2(v):
    v = np.asarray(v, np.float64); return float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()) + 1 / (len(v) + 1)))


def wilson(k, n, z=1.959964):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


# ------------------------------------------------------------------------------------------------ jets, library, edits
def indices(per_class, seed):
    from jetdistill.config import RESULTS
    y = np.asarray(np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')['y']); rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per_class, replace=False) for c in range(5)]))


def library(x):
    from interp.e11_mars_all import handbuilt
    H = handbuilt(x); pt = np.asarray(x[..., 0], np.float64); real = pt > 0
    H['H:ΣpT[all]'] = pt.sum(1); H['H:N[all]'] = real.sum(1).astype(np.float64)
    return H


def window_of(key):
    m = re.search(r'\[([^\]]+)\]$', key)
    from interp.e11_mars_all import WINDOWS
    return m.group(1) if (m and m.group(1) in WINDOWS) else 'all'


def resort(x):
    o = np.argsort(-x[..., 0], axis=1, kind='stable'); return np.take_along_axis(x, o[..., None], axis=1)


def edit(x, kind, w):
    from interp.e11_mars_all import WINDOWS
    lo, hi = WINDOWS[w]; e = x.copy(); real = e[..., 0] > 0; m = np.zeros(real.shape, bool); m[:, lo:hi] = True; m &= real
    if kind == 'pt':
        e[..., 0] = np.where(m, e[..., 0] * np.float32(EDIT_K), e[..., 0]); return resort(e)
    for c in (1, 2): e[..., c] = np.where(m, e[..., c] * np.float32(EDIT_K), e[..., c])
    return e


def rotate(x, rng):
    th = rng.uniform(0, 2 * np.pi, len(x)); e = x.copy(); real = e[..., 0] > 0
    c, s = np.cos(th).astype(np.float32)[:, None], np.sin(th).astype(np.float32)[:, None]
    e[..., 1] = np.where(real, c * x[..., 1] - s * x[..., 2], x[..., 1]); e[..., 2] = np.where(real, s * x[..., 1] + c * x[..., 2], x[..., 2])
    return e


def in_range(net, x):
    """JEDI-linear l1 input quantizer range (as interp.e5_causal_jedi.in_range)"""
    q = net.L['l1'].iq.quantizer
    k_, i_, f_ = (np.asarray(getattr(q, a)).reshape(64, 3) for a in ('k', 'i', 'f'))
    hi = 2.0 ** i_ - 2.0 ** -f_; lo = np.where(k_ > 0, -2.0 ** i_, 0.0)
    xn = np.asarray(net.inputs(x, 64)); real = x[..., 0] > 0
    return ~((((xn > hi[None]) | (xn < lo[None])) & real[..., None]).any((1, 2)))


# ------------------------------------------------------------------------------------------------ models
class FloatDS:
    """dsets-d64 float forward in numpy (as interp.x5_dsets_sparsity.FloatDS)"""
    def __init__(self, path):
        z = np.load(path); self.z = {k: np.asarray(z[k], np.float64) for k in z.files}
        self.shift, self.scale = self.z['shift'], self.z['scale']

    def run(self, x, batch=2500):
        out = dict(S=[], logits=[]); z = self.z
        for i in range(0, len(x), batch):
            xb = np.asarray(x[i:i + batch], np.float64); m = (xb[..., 0] > 0).astype(np.float64)
            xn = ((xb - self.shift) / self.scale) * m[..., None]
            a = np.maximum(xn @ z['phi_0_weight'].T + z['phi_0_bias'], 0)
            u = np.maximum(a @ z['phi_2_weight'].T + z['phi_2_bias'], 0) * m[..., None]; S = u.sum(1)
            h = np.maximum(S @ z['rho_0_weight'].T + z['rho_0_bias'], 0); h = np.maximum(h @ z['rho_2_weight'].T + z['rho_2_bias'], 0)
            h = np.maximum(h @ z['rho_4_weight'].T + z['rho_4_bias'], 0)
            out['S'].append(S); out['logits'].append(h @ z['head_weight'].T + z['head_bias'])
        return {k: np.concatenate(v) for k, v in out.items()}


def get_model(name):
    """-> (run(x) -> {layer: (J, w), 'logits'}, layers, reference accuracy)"""
    if name == 'jedi':
        from interp.jedi_core import Jedi
        net = Jedi(64); return (lambda x: net.run(x, keep=('S', 'h4', 'logits'))), ['S', 'h4'], 0.809
    if name == 'dsets':
        small = Path(os.environ.get('AI_SMALL_MODELS', '/j-jepa-vol/aaron-interp/models/small')) / 'dsets-d64-n64' / 'seed0'
        net = FloatDS(small / 'weights.npz'); acc = json.loads((small / 'test.json').read_text())['accuracy']
        return net.run, ['S'], acc
    from interp.x9_efn_pfn import Flow
    net = Flow('pfn_64-64-32_s0')
    return (lambda x: net.run(x, keep=('L', 'logits'))), ['L'], net.info['acc_test']


# ------------------------------------------------------------------------------------------------ MARS-style naming fit
class Fitter:
    """forward selection as jetdistill.mars.fit_neuron (same candidates, knots, products), with incremental orthogonalisation
    (the gain of column c given the chosen basis Q: (c·r)² / (|c|² − |Qᵀc|²), r ⟂ Q); linear target, <= MAX_TERMS terms"""

    def __init__(self, Htr, Hdev, keys, n_sel, seed=1):
        from jetdistill import mars
        self.mars = mars; self.keys = keys; self.Htr, self.Hdev = Htr, Hdev; Ntr = len(Htr[keys[0]])
        self.sel = np.sort(np.random.default_rng(seed).choice(Ntr, min(n_sel, Ntr), replace=False))
        self.knots = {k: mars.thresholds(Htr[k], False, False) for k in keys}
        cols, desc, self.obs_cols = [], [], {}
        for k in keys:
            v = Htr[k][self.sel]; i0 = len(desc)
            cols.append(v); desc.append(dict(q=k, kind='lin', t=None))
            for t in self.knots[k]: cols.append(np.maximum(0, v - t)); desc.append(dict(q=k, kind='gt', t=float(t)))
            for t in self.knots[k]: cols.append(np.maximum(0, t - v)); desc.append(dict(q=k, kind='lt', t=float(t)))
            self.obs_cols[k] = (i0 + 1, len(desc))                     # the hinge columns of k
        self.C = np.stack(cols, 1); self.desc = desc; self.Cn2 = (self.C ** 2).sum(0)
        Z = np.stack([Htr[k][self.sel] for k in keys], 1); self.Zs = (Z - Z.mean(0)) / np.maximum(Z.std(0), 1e-12)
        log(f'fitter: {len(keys)} observables, {self.C.shape[1]} candidate columns, {len(self.sel)} selection jets')

    def tv(self, tm, H): return self.mars.term_value(tm, H)

    def fit(self, y_tr, y_dev):
        n = len(self.sel); ys = y_tr[self.sel]; C = self.C
        q0 = np.full(n, 1 / np.sqrt(n)); Qb = [q0]; r = ys - q0 * (q0 @ ys); proj = (q0 @ C) ** 2
        chosen, path, coefs = [], [], []
        Btr, Bdev = [np.ones(len(y_tr))], [np.ones(len(y_dev))]; vdev = max(y_dev.var(), 1e-24)
        for _ in range(MAX_TERMS):
            den = self.Cn2 - proj; num = (C.T @ r) ** 2
            g = np.where(den > 1e-9 * np.maximum(self.Cn2, 1e-300), num / np.maximum(den, 1e-300), 0.0)
            i = int(g.argmax()); best = (float(g[i]), self.desc[i], C[:, i])
            if chosen:
                Qm = np.stack(Qb, 1); rn = np.linalg.norm(r)
                corr = np.abs(self.Zs.T @ r) / max(rn, 1e-300); top = [self.keys[j] for j in np.argsort(-corr)[:10]]
                for tm in chosen:
                    if tm.get('q2') is not None or tm['kind'] == 'lin': continue
                    pv = self.tv(tm, {tm['q']: self.Htr[tm['q']][self.sel]})
                    for k in top:
                        if k == tm['q']: continue
                        a, b_ = self.obs_cols[k]; P = C[:, a:b_] * pv[:, None]; Pp = P - Qm @ (Qm.T @ P)
                        nn = (Pp ** 2).sum(0); gg = np.where(nn > 1e-9 * np.maximum((P ** 2).sum(0), 1e-300), (P.T @ r) ** 2 / np.maximum(nn, 1e-300), 0.0)
                        j = int(gg.argmax())
                        if gg[j] > best[0]:
                            d = self.desc[a + j]; best = (float(gg[j]), dict(q=tm['q'], kind=tm['kind'], t=tm['t'], q2=k, kind2=d['kind'], t2=d['t']), P[:, j])
            gbest, tm, v = best
            if gbest <= 0: break
            Qm = np.stack(Qb, 1); v = v - Qm @ (Qm.T @ v); v = v - Qm @ (Qm.T @ v); nv = np.linalg.norm(v)
            if nv < 1e-12: break
            q = v / nv; Qb.append(q); proj = proj + (q @ C) ** 2; r = r - q * (q @ r)
            chosen.append(tm); Btr.append(self.tv(tm, self.Htr)); Bdev.append(self.tv(tm, self.Hdev))
            c = np.linalg.lstsq(np.stack(Btr, 1), y_tr, rcond=None)[0]; coefs.append(c)
            r2 = 1 - ((y_dev - np.stack(Bdev, 1) @ c) ** 2).mean() / vdev; path.append(float(r2))
            if r2 > 0.9995: break
        if not chosen: return dict(terms=[], coef=[float(y_tr.mean())], K=0, dev_path=[])
        mx = max(path); K = next(k + 1 for k, p in enumerate(path) if p >= mx - 0.005)
        return dict(terms=chosen[:K], coef=[float(v) for v in coefs[K - 1]], K=K, dev_path=path)

    def predict(self, F, H):
        v = np.full(len(H[self.keys[0]]), F['coef'][0])
        for t, c in zip(F['terms'], F['coef'][1:]): v = v + c * self.tv(t, H)
        return v


def atoms(F):
    a = lambda k: 1 if k == 'lin' else 2
    return sum(a(t['kind']) + (a(t['kind2']) if t.get('q2') else 0) for t in F['terms'])


def term_str(t):
    def one(q, k, th): return q if k == 'lin' else (f'max(0, {q} − {th:.4g})' if k == 'gt' else f'max(0, {th:.4g} − {q})')
    s = one(t['q'], t['kind'], t['t'])
    return s + (' × ' + one(t['q2'], t['kind2'], t['t2']) if t.get('q2') else '')


# ------------------------------------------------------------------------------------------------ SAE
def train_sae(X, Xdev, m, lam, seed, steps=3000, batch=512, lr=1e-3):
    import jax, jax.numpy as jnp
    d = X.shape[1]; key = jax.random.PRNGKey(seed)
    Wd = jax.random.normal(key, (m, d)); Wd = Wd / jnp.linalg.norm(Wd, axis=1, keepdims=True)
    p = dict(We=Wd.T * 1.0, be=jnp.zeros(m), Wd=Wd, bd=jnp.zeros(d))

    def enc(p, x): return jax.nn.relu((x - p['bd']) @ p['We'] + p['be'])

    def loss(p, x):
        f = enc(p, x); xh = f @ p['Wd'] + p['bd']
        return ((x - xh) ** 2).sum(1).mean() / d + lam * f.sum(1).mean()

    g_ = jax.jit(jax.value_and_grad(loss)); b1, b2, eps = 0.9, 0.999, 1e-8
    mo = {k: jnp.zeros_like(v) for k, v in p.items()}; ve = {k: jnp.zeros_like(v) for k, v in p.items()}

    @jax.jit
    def step(p, mo, ve, x, t):
        l, g = g_(p, x)
        mo = {k: b1 * mo[k] + (1 - b1) * g[k] for k in p}; ve = {k: b2 * ve[k] + (1 - b2) * g[k] ** 2 for k in p}
        p = {k: p[k] - lr * (mo[k] / (1 - b1 ** t)) / (jnp.sqrt(ve[k] / (1 - b2 ** t)) + eps) for k in p}
        p['Wd'] = p['Wd'] / jnp.maximum(jnp.linalg.norm(p['Wd'], axis=1, keepdims=True), 1e-8)
        return p, mo, ve, l
    rng = np.random.default_rng(seed); Xj = jnp.asarray(X, jnp.float32)
    for t in range(1, steps + 1):
        idx = rng.integers(0, len(X), batch); p, mo, ve, l = step(p, mo, ve, Xj[idx], t)
    P = {k: np.asarray(v, np.float64) for k, v in p.items()}
    encn = lambda x: np.maximum((x - P['bd']) @ P['We'] + P['be'], 0)
    fd = encn(Xdev); xh = fd @ P['Wd'] + P['bd']
    fvu = float(((Xdev - xh) ** 2).sum() / ((Xdev - Xdev.mean(0)) ** 2).sum())
    return P, encn, fvu, float((fd > 0).sum(1).mean())


# ------------------------------------------------------------------------------------------------ bootstrap R²
def boot_r2(W, Y, F):
    """W (B, N) resample counts; Y, F (N, D) -> (B, D) R² of F against Y on each resample"""
    N = Y.shape[0]; mu = Y.mean(0); Yc = Y - mu; E2 = (Y - F) ** 2
    s1 = W @ Yc / N; s2 = W @ (Yc ** 2) / N; se = W @ E2 / N
    return 1 - se / np.maximum(s2 - s1 ** 2, 1e-300)


# ------------------------------------------------------------------------------------------------ main
def main(a):
    smoke = a.smoke; sfx = '_smoke' if smoke else ''
    global N_ROT, N_ROT_STD, B, BC, N_CAUSAL
    pc_fit, pc_test = (600, 400) if smoke else (5000, 5000)
    if smoke: N_ROT, N_ROT_STD, B, BC, N_CAUSAL = 3, 2, 50, 50, 600
    pre_path = OUT / 'x11_prereg.json'
    if not pre_path.exists(): pre_path.write_text(json.dumps(PRE, indent=1, ensure_ascii=False)); log(f'wrote {pre_path}')
    elif json.loads(pre_path.read_text()) != PRE: raise SystemExit('x11_prereg.json on the volume differs from interp/x11_prereg.json')
    else: log('preregistration unchanged')
    from interp.jedi_core import sample, Jedi
    R = dict(model=a.model, prereg=str(pre_path), smoke=smoke, layers={})
    x, y, _ = sample(pc_fit, 0); xt, yt, _ = sample(pc_test, 29)
    i0, i1 = indices(pc_fit, 0), indices(pc_test, 29); keep = ~np.isin(i1, i0); xt, yt = xt[keep], yt[keep]
    J = len(x); perm = np.random.default_rng(0).permutation(J); nf = 2 * J // 3; itr, idv = perm[:nf], perm[nf:]
    srt = float((np.diff(x[..., 0], axis=1) <= 1e-6).all(1).mean())
    log(f'jets: fit {J} (train {nf}, dev {J - nf}), test {len(xt)} (removed {int((~keep).sum())} overlapping); fraction of fit jets pT-sorted {srt:.4f}')
    R['jets'] = dict(fit=J, fit_train=nf, fit_dev=J - nf, test=len(xt), test_overlap_removed=int((~keep).sum()), pt_sorted=srt)

    # causal jets: in JEDI-linear's input range under every edit (same jets for all models)
    from interp.e11_mars_all import WINDOWS
    EDITS = [(k, w) for w in WINDOWS for k in ('pt', 'r')]
    jn = Jedi(64); rng = np.random.default_rng(5); cand = rng.permutation(len(xt))
    ok = in_range(jn, xt[cand])
    for k, w in EDITS: ok &= in_range(jn, edit(xt[cand], k, w))
    ci_ = cand[ok][:N_CAUSAL]; xc = xt[ci_]
    log(f'causal jets: {len(ci_)} of {len(cand)} candidates in range under all {len(EDITS)} edits ({ok.mean():.3f} in range)')
    R['jets'].update(causal=len(ci_), causal_in_range_fraction=float(ok.mean()))
    del jn
    xe = {f'{k}:{w}': edit(xc, k, w) for k, w in EDITS}; xe['rot'] = rotate(xc, np.random.default_rng(19))

    # library
    H = library(x); Ht = library(xt); Hc = library(xc); He = {e: library(v) for e, v in xe.items()}
    keys = [k for k in H if np.isfinite(H[k]).all() and np.isfinite(Ht[k]).all() and np.ptp(H[k][itr]) > 0]
    rot_dmax = max(float(np.abs(He['rot'][k] - Hc[k]).max() / max(H[k].std(), 1e-12)) for k in keys)
    log(f'library: {len(keys)} observables; rotation edit max |Δ observable| / s.d. = {rot_dmax:.2e}')
    R['library'] = dict(n=len(keys), keys=keys, rotation_max_change_sd=rot_dmax)
    Htr = {k: H[k][itr] for k in keys}; Hdv = {k: H[k][idv] for k in keys}
    fitter = Fitter(Htr, Hdv, keys, N_SEL if not smoke else 1500)

    # model forward
    run, layers, acc_ref = get_model(a.model)
    A = run(x); At = run(xt); Ac = run(xc); Ae = {e: run(v) for e, v in xe.items()}
    acc = float((A['logits'].argmax(1) == y).mean()); acct = float((At['logits'].argmax(1) == yt).mean())
    log(f'{a.model}: accuracy fit sample {acc:.4f}, test sample {acct:.4f} (reference {acc_ref:.4f})')
    R['accuracy'] = dict(fit=acc, test=acct, reference=acc_ref)

    W = np.stack([np.bincount(r_, minlength=len(xt)) for r_ in np.random.default_rng(11).integers(0, len(xt), (B, len(xt)))]).astype(np.float64)
    BIc = np.random.default_rng(13).integers(0, len(xc), (BC, len(xc)))

    for L in layers:
        h, ht, hc = A[L], At[L], Ac[L]; he = {e: Ae[e][L] for e in xe}
        htr = h[itr]
        live = np.flatnonzero((htr.std(0) > 1e-6) & ((htr > 1e-6).mean(0) >= 0.005)); d = len(live)
        log(f'--- layer {L}: {d} live units of {h.shape[1]}: {live.tolist()}')
        mu = htr[:, live].mean(0); sd = htr[:, live].std(0); s_rms = float(np.sqrt((sd ** 2).mean()))
        # bases: name -> list of instances; instance = (label, transform(acts (J, w)) -> (J, D))
        bases = {'native': [('native', lambda Z: Z[:, live])]}
        def rot(seed):
            M = np.random.default_rng(seed).standard_normal((d, d)); Q, Rr = np.linalg.qr(M); return Q * np.sign(np.diag(Rr))
        bases['rot_raw'] = [(f'rot_raw{i}', (lambda Q: lambda Z: Z[:, live] @ Q.T)(rot(100 + i))) for i in range(N_ROT)]
        bases['rot_std'] = [(f'rot_std{i}', (lambda Q: lambda Z: ((Z[:, live] - mu) / sd) @ Q.T)(rot(200 + i))) for i in range(N_ROT_STD)]
        ev, V = np.linalg.eigh(np.cov(htr[:, live].T)); V = V[:, ::-1]
        bases['pca'] = [('pca', (lambda V: lambda Z: (Z[:, live] - mu) @ V)(V))]
        sae_info = {}
        Xs = (htr[:, live] - mu) / s_rms; Xsd = (h[idv][:, live] - mu) / s_rms
        for mult in (2, 4):
            m = mult * d; trial = []
            for lam in LAMBDAS:
                P, encn, fvu, l0 = train_sae(Xs, Xsd, m, lam, seed=mult, steps=300 if smoke else 3000)
                trial.append(dict(lam=lam, fvu=fvu, l0=l0)); log(f'SAE {mult}x (m {m}) λ {lam}: dev FVU {fvu:.4f}, mean L0 {l0:.2f}')
                if fvu <= 0.10: break
            enc = (lambda e: lambda Z: e((Z[:, live] - mu) / s_rms))(encn)
            f = enc(htr); alive = np.flatnonzero((f > 1e-6).mean(0) >= 0.005); energy = (f ** 2).mean(0)
            topk = [int(i) for i in np.argsort(-energy) if i in set(alive.tolist())][:d]
            sae_info[f'sae{mult}x'] = dict(m=m, lambda_trials=trial, lam=trial[-1]['lam'], fvu=trial[-1]['fvu'], l0=trial[-1]['l0'],
                                           alive=len(alive), topk=topk, chosen_by_rule=trial[-1]['fvu'] <= 0.10)
            bases[f'sae{mult}x_all'] = [(f'sae{mult}x_all', (lambda e, idx: lambda Z: e(Z)[:, idx])(enc, alive))]
            sae_info[f'sae{mult}x']['alive_idx'] = alive.tolist()
        RL = dict(live=live.tolist(), d=d, sae=sae_info, bases={}); RBC = {}

        for bname, insts in bases.items():
            for label, T in insts:
                t0 = time.time()
                Ytr, Ydv, Yt, Yc = T(htr), T(h[idv]), T(ht), T(hc); Ye = {e: T(he[e]) for e in xe}
                D = Yt.shape[1]; Ft = np.zeros_like(Yt); dirs = []
                for j in range(D):
                    if np.ptp(Ytr[:, j]) < 1e-9:
                        dirs.append(dict(j=j, constant=True, r2=0.0, K=0, atoms=0)); Ft[:, j] = Ytr[:, j].mean(); continue
                    F = fitter.fit(Ytr[:, j], Ydv[:, j]); Ft[:, j] = fitter.predict(F, Ht)
                    r2 = float(1 - ((Yt[:, j] - Ft[:, j]) ** 2).mean() / max(Yt[:, j].var(), 1e-300))
                    sp = [abs(c) * float(fitter.tv(t, Htr).std()) for t, c in zip(F['terms'], F['coef'][1:])]
                    top = F['terms'][int(np.argmax(sp))] if sp else None
                    dirs.append(dict(j=j, r2=r2, K=F['K'], atoms=atoms(F), dev_path=F['dev_path'], F=F, top=term_str(top) if top else None,
                                     top_q=top['q'] if top else None, terms=[term_str(t) for t in F['terms']]))
                Rb = boot_r2(W, Yt, Ft)
                pn = (1 + (Rb < R2_NAMED).sum(0)) / (B + 1); pn_bh = bh(pn)
                for jj, dd in enumerate(dirs): dd['p_named'] = float(pn[jj]); dd['bh_named'] = bool(pn_bh[jj] < 0.05)
                named = [dd for dd in dirs if dd['r2'] >= R2_NAMED]
                # causal test
                sdy = Yt.std(0)
                for dd in named:
                    j = dd['j']; w = window_of(dd['top_q']); best = None
                    for kind in ('pt', 'r'):
                        e = f'{kind}:{w}'; dp = fitter.predict(dd['F'], He[e]) - fitter.predict(dd['F'], Hc)
                        sz = float(np.abs(dp).mean() / max(sdy[j], 1e-300))
                        if best is None or sz > best[1]: best = (e, sz, dp)
                    e, sz, dp = best; da = Ye[e][:, j] - Yc[:, j]; dd['edit'] = e; dd['pred_size_sd'] = sz
                    dd['rot_size_sd'] = float(np.abs(Ye['rot'][:, j] - Yc[:, j]).mean() / max(sdy[j], 1e-300))
                    dd['edit_size_sd'] = float(np.abs(da).mean() / max(sdy[j], 1e-300))
                    if sz < MIN_PRED: dd['causal'] = 'untestable'; continue
                    vp = dp.var()
                    sl = float(np.cov(dp, da)[0, 1] / vp) if vp > 0 else np.nan
                    P_, A_ = dp[BIc], da[BIc]
                    slb = ((P_ - P_.mean(1, keepdims=True)) * (A_ - A_.mean(1, keepdims=True))).mean(1) / np.maximum(P_.var(1), 1e-300)
                    c = ci(slb); dd['slope'] = sl; dd['slope_ci'] = c
                    dd['causal'] = 'pass' if (c[0] >= SLOPE_LO and c[1] <= SLOPE_HI) else 'fail'
                    dd['p_equiv'] = float(max((1 + (slb <= SLOPE_LO).sum()) / (BC + 1), (1 + (slb >= SLOPE_HI).sum()) / (BC + 1)))
                    dd['corr'] = float(np.corrcoef(dp, da)[0, 1]) if dp.std() > 0 and da.std() > 0 else None
                tested = [dd for dd in named if dd.get('causal') in ('pass', 'fail')]
                pe = bh([dd['p_equiv'] for dd in tested])
                for dd, q in zip(tested, pe): dd['bh_equiv'] = bool(q < 0.05)
                # mismatched-formula control
                mis = []
                if len(tested) >= 2:
                    rr = np.random.default_rng(17); n_ = len(tested)
                    while True:
                        pm = rr.permutation(n_)
                        if (pm != np.arange(n_)).all(): break
                    for dd, k2 in zip(tested, pm):
                        o = tested[k2]; e = dd['edit']; dp = fitter.predict(o['F'], He[e]) - fitter.predict(o['F'], Hc)
                        da = Ye[e][:, dd['j']] - Yc[:, dd['j']]; P_, A_ = dp[BIc], da[BIc]
                        if dp.var() <= 0: mis.append(False); continue
                        slb = ((P_ - P_.mean(1, keepdims=True)) * (A_ - A_.mean(1, keepdims=True))).mean(1) / np.maximum(P_.var(1), 1e-300)
                        c = ci(slb); mis.append(bool(c[0] >= SLOPE_LO and c[1] <= SLOPE_HI))
                for dd in dirs: dd.pop('F', None)
                n_pass = sum(dd.get('causal') == 'pass' for dd in named)
                inst = summarise(label, dirs, Rb, dict(mismatched_pass=int(sum(mis)), mismatched_n=len(mis), seconds=round(time.time() - t0, 1)))
                if bname.startswith('sae'): RBC[label] = Rb
                RL['bases'].setdefault(bname, []).append(inst)
                log(f'{L} {label}: D {D}, named {len(named)} ({len(named) / D:.3f} [{inst["frac_named_ci"][0]:.3f}, {inst["frac_named_ci"][1]:.3f}]), '
                    f'BH {inst["n_named_bh"]}; causal pass {n_pass} / tested {len(tested)} (untestable {inst["n_untestable"]}; mismatched control '
                    f'{sum(mis)}/{len(mis)}); mean K {inst["mean_K_named"]}, median R² {inst["median_r2"]:.3f}; {time.time() - t0:.0f} s')
            R['layers'][L] = RL; (OUT / f'x11_{a.model}{sfx}.json').write_text(json.dumps(clean(R), ensure_ascii=False))
        # top-k SAE instances: subsets of the all-alive SAE directions (same fits, same bootstrap)
        for mult in (2, 4):
            al = RL['sae'][f'sae{mult}x']['alive_idx']; tk = RL['sae'][f'sae{mult}x']['topk']; src = RL['bases'][f'sae{mult}x_all'][0]
            pos = [al.index(i) for i in tk]; RL['bases'][f'sae{mult}x'] = [subset_instance(src, pos, RBC[f'sae{mult}x_all'], f'sae{mult}x_top{len(tk)}')]
            log(f'{L} sae{mult}x top-{len(tk)}: named {RL["bases"][f"sae{mult}x"][0]["n_named"]}, causal pass {RL["bases"][f"sae{mult}x"][0]["n_causal_pass"]}')
        RL['stats'] = stats_layer(RL); log(f'{L} stats: ' + json.dumps(clean({k: dict(diff=v.get('diff'), ci=v.get('ci'), p=v.get('p')) for k, v in RL['stats'].items() if isinstance(v, dict)})))
        R['layers'][L] = RL; (OUT / f'x11_{a.model}{sfx}.json').write_text(json.dumps(clean(R), ensure_ascii=False))
    R['seconds'] = time.time() - T0
    (OUT / f'x11_{a.model}{sfx}.json').write_text(json.dumps(clean(R), ensure_ascii=False)); log(f'done; wrote x11_{a.model}{sfx}.json')


def summarise(label, dirs, Rb, extra=None):
    named = [dd for dd in dirs if dd['r2'] >= R2_NAMED]; tested = [dd for dd in named if dd.get('causal') in ('pass', 'fail')]
    D = len(dirs); n_pass = sum(dd.get('causal') == 'pass' for dd in named); fr_b = (Rb >= R2_NAMED).mean(1)
    out = dict(label=label, D=D, frac_named=len(named) / max(D, 1), frac_named_boot=fr_b.tolist(), frac_named_ci=ci(fr_b),
               frac_named_wilson=wilson(len(named), D), n_named=len(named), n_named_bh=int(sum(dd.get('bh_named', False) for dd in dirs)),
               n_causal_pass=n_pass, frac_causal=n_pass / max(D, 1), frac_causal_wilson=wilson(n_pass, D),
               frac_causal_of_named=n_pass / len(named) if named else None,
               n_untestable=sum(dd.get('causal') == 'untestable' for dd in named), n_tested=len(tested),
               n_equiv_bh=int(sum(dd.get('bh_equiv', False) for dd in tested)),
               mean_K_named=float(np.mean([dd['K'] for dd in named])) if named else None,
               mean_atoms_named=float(np.mean([dd['atoms'] for dd in named])) if named else None,
               median_r2=float(np.median([dd['r2'] for dd in dirs])) if dirs else None,
               median_rot_over_edit=float(np.median([dd['rot_size_sd'] / max(dd['edit_size_sd'], 1e-12) for dd in named])) if named else None,
               directions=dirs)
    if extra: out.update(extra)
    return out


def subset_instance(src, pos, Rb, label):
    return summarise(label, [src['directions'][p] for p in pos], Rb[:, pos], dict(positions=pos))


def stats_layer(RL):
    """native vs rotations / PCA / SAE, paired over the same bootstrap resamples"""
    Bs = RL['bases']; nat = Bs['native'][0]; nb = np.array(nat['frac_named_boot']); S = {}
    for rb in ('rot_raw', 'rot_std'):
        I = Bs[rb]; fr = np.array([i['frac_named'] for i in I]); fc = np.array([i['frac_causal'] for i in I])
        mb = np.mean([i['frac_named_boot'] for i in I], 0); dlt = nb - mb
        S[f'native_minus_{rb}'] = dict(diff=nat['frac_named'] - fr.mean(), ci=ci(dlt), p=float((1 + (fr >= nat['frac_named']).sum()) / (len(fr) + 1)),
                                       p_kind='permutation over rotations (1 + #{rot >= native}) / (n + 1)', rot_fracs=fr.tolist(),
                                       rot_min=float(fr.min()), rot_max=float(fr.max()), native=nat['frac_named'], boot_p=boot_p2(dlt))
        S[f'causal_native_minus_{rb}'] = dict(diff=nat['frac_causal'] - fc.mean(), p=float((1 + (fc >= nat['frac_causal']).sum()) / (len(fc) + 1)),
                                              rot_fracs=fc.tolist(), native=nat['frac_causal'])
        S[f'K_{rb}'] = dict(native=nat['mean_K_named'], rot=[i['mean_K_named'] for i in I])
    for b in ('pca', 'sae2x', 'sae4x', 'sae2x_all', 'sae4x_all'):
        i = Bs[b][0]; dlt = np.array(i['frac_named_boot']) - nb
        S[f'{b}_minus_native'] = dict(diff=i['frac_named'] - nat['frac_named'], ci=ci(dlt), p=boot_p2(dlt), other=i['frac_named'], native=nat['frac_named'],
                                      causal_other=i['frac_causal'], causal_native=nat['frac_causal'])
    return S


# ------------------------------------------------------------------------------------------------ combine + figure
def combine():
    M = {m: json.loads((OUT / f'x11_{m}.json').read_text()) for m in ('jedi', 'dsets', 'pfn') if (OUT / f'x11_{m}.json').exists()}
    rows = [(m, L) for m in M for L in M[m]['layers']]
    S = dict(models=list(M), accuracy={m: M[m]['accuracy'] for m in M}, table={}, headline=[], cross={})
    for m, L in rows:
        RL = M[m]['layers'][L]; Bs = RL['bases']; st = RL['stats']
        S['table'][f'{m}:{L}'] = dict(d=RL['d'], **{b: dict(frac_named=Bs[b][0]['frac_named'], ci=Bs[b][0]['frac_named_ci'], frac_causal=Bs[b][0]['frac_causal'],
                                                         mean_K=Bs[b][0]['mean_K_named'], n_named=Bs[b][0]['n_named'], D=Bs[b][0]['D'])
                                                    for b in ('native', 'pca', 'sae2x', 'sae4x', 'sae2x_all', 'sae4x_all')},
                                      rot_raw=dict(fracs=[i['frac_named'] for i in Bs['rot_raw']], causal=[i['frac_causal'] for i in Bs['rot_raw']]),
                                      rot_std=dict(fracs=[i['frac_named'] for i in Bs['rot_std']]), sae=RL['sae'], stats=st)
        for k in ('native_minus_rot_raw', 'sae2x_minus_native', 'sae4x_minus_native'):
            S['headline'].append(dict(test=f'{m}:{L} {k}', diff=st[k]['diff'], ci=st[k].get('ci'), p=st[k]['p']))
    if 'jedi' in M:
        nj = np.array(M['jedi']['layers']['S']['bases']['native'][0]['frac_named_boot'])
        for m, L in (('dsets', 'S'), ('pfn', 'L')):
            if m not in M: continue
            nb = np.array(M[m]['layers'][L]['bases']['native'][0]['frac_named_boot']); dlt = nj - nb
            d0 = M['jedi']['layers']['S']['bases']['native'][0]['frac_named'] - M[m]['layers'][L]['bases']['native'][0]['frac_named']
            S['cross'][f'jedi:S − {m}:{L} native'] = dict(diff=d0, ci=ci(dlt), p=boot_p2(dlt))
            S['headline'].append(dict(test=f'jedi:S − {m}:{L} native', diff=d0, ci=ci(dlt), p=boot_p2(dlt)))
    hp = holm([h['p'] for h in S['headline']])
    for h, q in zip(S['headline'], hp): h['p_holm'] = float(q)
    (OUT / 'x11_summary.json').write_text(json.dumps(clean(S), indent=1, ensure_ascii=False)); log('wrote x11_summary.json')
    figure(M, rows)


def figure(M, rows):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    names = {'jedi:S': 'JEDI-linear S\n(pooled objects)', 'jedi:h4': 'JEDI-linear h4\n(first MLP layer)', 'dsets:S': 'Deep Sets S\n(pooled latent)', 'pfn:L': 'PFN latent'}
    fig, axs = plt.subplots(1, 2, figsize=(11, 0.9 + 1.0 * len(rows)), sharey=True)
    col = dict(native='#1f4e9c', pca='#7a7a7a', sae2x='#d9822b', sae4x='#a3461a', rot='#9bb7d4')
    for ax, key, ttl in ((axs[0], 'frac_named', 'named: held-out R² ≥ 0.9 with ≤ 8 hinge terms'), (axs[1], 'frac_causal', 'named and edit slope 95 % CI inside [0.8, 1.2]')):
        for yi, (m, L) in enumerate(rows):
            Bs = M[m]['layers'][L]['bases']; yy = len(rows) - 1 - yi
            rf = [i[key] for i in Bs['rot_raw']]
            ax.scatter(rf, yy + np.random.default_rng(0).uniform(-0.12, 0.12, len(rf)), s=12, color=col['rot'], label='20 random rotations' if yi == 0 else None, zorder=2)
            for b, mk, off in (('native', 'o', 0.0), ('pca', 'D', 0.0), ('sae2x', 's', 0.0), ('sae4x', '^', 0.0)):
                ax.scatter([Bs[b][0][key]], [yy + off], s=70 if b == 'native' else 45, marker=mk, color=col[b], zorder=3,
                           label={'native': 'native units', 'pca': 'PCA directions', 'sae2x': 'SAE 2× (top d)', 'sae4x': 'SAE 4× (top d)'}[b] if yi == 0 else None)
            if key == 'frac_named':
                c = Bs['native'][0]['frac_named_ci']; ax.plot(c, [yy, yy], color=col['native'], lw=2, zorder=3)
        ax.set_xlim(-0.03, 1.03); ax.set_title(ttl, fontsize=10, loc='left'); ax.set_xlabel('fraction of directions')
        ax.grid(axis='x', alpha=0.3)
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    axs[0].set_yticks(range(len(rows))); axs[0].set_yticklabels([names.get(f'{m}:{L}', f'{m}:{L}') + f'\nd = {M[m]["layers"][L]["d"]}' for m, L in rows][::-1], fontsize=9)
    axs[0].legend(fontsize=8, loc='lower right', frameon=False)
    fig.suptitle('X11: share of directions named as (pT, r) rank-window formulas, by basis (test jets, N = 25,000 minus overlap)', fontsize=10.5, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.94)); p = OUT / 'x11_atlas.png'; fig.savefig(p, dpi=150); log(f'wrote {p}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--model', choices=['jedi', 'dsets', 'pfn']); ap.add_argument('--combine', action='store_true')
    ap.add_argument('--smoke', action='store_true'); a = ap.parse_args(); log('args', vars(a))
    if a.combine: combine()
    else: main(a)
