"""E25w (claims A1f, A4, A9; exploratory, second official family): in the official 16-feature JEDI-linear n64 (reads angles only
through ΔR), (1) at which stage the axis-relative mass variables are held in a low-dimensional causal subspace, (2) the full call
transition under a translation dose, (3) whether the translation response exceeds non-constant baseline networks, (4) n32 / n128.
   python -m interp.e25w_jedi16_mass --part {form,dose,base} [--smoke]   -> $AI_RESULTS/jedi/e25w_<part>[_smoke].json

Sample: rows of interp.jedi_core.sample(1500, 2521) minus the rows of sample(3000, 2501) (E25 main), sample(1500, 2502) (E25
fresh) and sample(1500, 2507) (E25v). All 16 jetConstituentList columns from val_c16_f16.npy at those rows (first 64 slots; 128
for n128); abort unless the pt / etarel / phirel columns equal sample() x. Fit / test halves: rng(2522).permutation (not first-k).
Variables (stored axis, real slots): M0 = Σ pT, M2 = Σ pT ΔR² (ΔR column), D = ½ log(M0 M2) (= log m for small-angle massless).
Edits (all 16 columns recomputed from (pT, η, φ, Δη, Δφ): px py pz e erel pt ptrel η Δη φ Δφ ΔR cosθ by F(edited) - F(unedited);
etarot, phirot, costhetarel multiplied by the angle scale; E_J, pT_J fixed except under a pT scale, where they scale with pT):
  T_d    rigid translation Δη += d of every real particle (rotated-frame columns unchanged).
  R_d    rotation of (Δη, Δφ) about the stored axis by the angle that moves a particle at the pT-weighted mean r by d (random sign).
  dilR_d ΔR column only × s_d, s_d = sqrt(M2(T_d) / M2)  (the "ΔR × k" matched dilation: same M2 change as T_d, nothing else moved).
  dil_d  all angle columns scaled by s_d about the axis (a physical dilation with the same M2 change).
  do(M0) pT × c (whole jet; ptrel, erel unchanged), angles × c^-1/2 (M2 unchanged), c = M0_s / M0_b.
  do(M2) angles × sqrt(M2_s / M2_b) (M0 unchanged).          do(D) pT × sqrt(M0_s M2_s / (M0_b M2_b)) (angles unchanged).
Part form (16f n64 and 16f-perminv n64): 3,000 ordered training pairs (base, source) from the fit half (rng 2523; last 20 % for
  model selection), 2,000 test pairs from the test half (rng 2524). Counterfactual label = the network's exact call on the base jet
  after do(V) (V := source value). DAS at S, h4, h5, z, logits (k = 1, 2): h' = clip(h_b + (h_s - h_b) Q Qᵀ), clip = next quantizer
  range widened to the base value (as X10); Q trained (Adam 0.01, 600 steps, straight-through through the quantizers; selection on
  balanced held-out agreement); evaluated with the exact fixed-point tail. Effective IIA = agreement on test pairs whose
  counterfactual call differs from the base call; 1,000-resample bootstrap over pairs. Controls: 500 random orthonormal k-subspaces
  (p_rot = (1 + #random >= DAS) / 501); full-stage swap (source call at S..logits; at P: P set to the source's P, call by the
  exact network); paired bootstrap DAS - full swap. Holm over stage x variable x k. Ridge read-out of log M0, log M2, D per stage
  (P, g, S, h4, h5, z, logits; fit half, alpha by 1/4 dev split; R² on the test half); calibration on the test half: read-out
  shift under dilR_0.06 / true shift (2 log s for log M2), and read-out shift under T_0.06 / under dilR_0.06 (bootstrap over jets).
Part dose: models 16f n64, 16f-perminv n64, 16f n32 (first 32 slots), 16f n128 (128 slots) (official forward, jedi_core.Jedi).
  For d in DOSES: T_d, dilR_d, dil_d, R_d. Per arm: 5x5 call transition (unedited -> edited call) on all jets and on true W jets,
  true-W new-call fractions (g, q, W, Z, t) with 2,000-resample bootstrap CI, W->Z flow, McNemar (Z indicator on true W) vs R_d,
  ratio flow(T_d) / flow(dilR_d) and / flow(dil_d) (paired bootstrap). Holm over model x dose x arm McNemar tests.
Part base (16f n64): non-constant baselines with a refit read-out (multinomial logistic regression of the d7 input
  iq(ReLU(z)) on the fit half; d7 replaced by the fit): trained_refit (trained body), init_s (l1 trained; l2, dd, l3, d4, d5, d6
  fused kernels redrawn Glorot-uniform at init scale, biases 0; s 0-4), gauss_s (same layers redrawn N(mean, sd) of the trained fused
  kernel / bias, the redraw_untrained scale; s 0-4), perm_s (X8 variant a: non-zero entries of every fused kernel and bias of l1..d6
  permuted within the layer, masks kept, trained quantizers kept; s 0-9). Qualify: test accuracy > 0.40 and > 1 distinct call.
  Metric (test half, true W jets): calibrated ratio = flow(T_0.06) / flow(dilR_0.06); also flows of T, dilR, R. H0: trained ratio
  inside the qualifying-baseline distribution; p_ratio = (1 + #{b: |ratio_b - 1| <= |ratio_tr - 1|}) / (1 + n) (undefined
  baseline ratios, dilation flow <= 0, count as farther from 1); p_flow = (1 + #{b: flow_T,b >= flow_T,tr}) / (1 + n).
  DAS (k = 2, 50 random subspaces, 400 steps) on every qualifying baseline at S, h4, h5, z with its own counterfactual labels.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
ap = argparse.ArgumentParser(); ap.add_argument('--part', required=True, choices=['form', 'dose', 'base']); ap.add_argument('--smoke', action='store_true')
ap.add_argument('--nrand', type=int, default=None, help='random subspaces per DAS cell in part form (default 500); run2 used 5000 so that Holm over 30 cells can fall below 0.05')
args = ap.parse_args() if __name__ == '__main__' else ap.parse_args(['--part', 'form', '--smoke'])
SM = args.smoke
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)
FEAT16 = ['px', 'py', 'pz', 'e', 'erel', 'pt', 'ptrel', 'eta', 'etarel', 'etarot', 'phi', 'phirel', 'phirot', 'deltaR', 'costheta',
          'costhetarel']
C = {k: i for i, k in enumerate(FEAT16)}
CL = ['g', 'q', 'W', 'Z', 't']; W_, Z_ = 2, 3
DOSES = (0.015, 0.03, 0.045, 0.06, 0.09, 0.12)
PER = 150 if SM else 1500
NB = 200 if SM else 2000
NRAND = args.nrand or (20 if SM else 500)
STEPS = 60 if SM else 600
NTR, NTE = (500, 400) if SM else (3000, 2000)
B = 2500
VARS = ('M0', 'M2', 'D')
from interp.e19_common import wilson, mcnemar, holm, boot_mean, boot_ratio  # noqa: E402


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.ndarray): return clean(o.tolist())
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_): return bool(o)
    return o


# ------------------------------------------------------------------------------------------------ sample
def rows(y, per, seed):
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per, replace=False) for c in range(5)]))


def load(nslot=64):
    from jetdistill.config import RESULTS, DATA
    from interp.jedi_core import sample
    y = np.asarray(np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')['y'])
    idx = rows(y, PER, 2521); x3, ys, _ = sample(PER, 2521)
    assert np.array_equal(ys, y[idx])
    X = np.asarray(np.load(Path(DATA) / 'val_c16_f16.npy', mmap_mode='r')[idx, :nslot]).astype(np.float32)
    d = float(np.abs(X[:, :64][..., [5, 8, 11]] - x3).max()); log(f'load nslot {nslot}: max |16f cols - sample x| {d:.3g}')
    if d > 1e-3: raise SystemExit('ABORT: 16-feature columns differ from sample() x')
    excl = np.concatenate([rows(y, 3000, 2501), rows(y, 1500, 2502), rows(y, 1500, 2507)])
    keep = ~np.isin(idx, excl)
    return X[keep], ys[keep], idx[keep], d


def halves(N):
    p = np.random.default_rng(2522).permutation(N); return np.sort(p[:N // 2]), np.sort(p[N // 2:])


# ------------------------------------------------------------------------------------------------ kinematics / edits
def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi


def jinfo(X):
    Xd = X.astype(np.float64); rl = Xd[..., C['pt']] > 0; pt = np.where(rl, Xd[..., C['pt']], 0.0)
    def jmed(v): return np.nanmedian(np.where(rl, v, np.nan), 1)  # noqa: E306
    ptJ = jmed(pt / np.where(rl, Xd[..., C['ptrel']], 1)); EJ = jmed(Xd[..., C['e']] / np.where(rl, Xd[..., C['erel']], 1))
    m2 = np.clip(Xd[..., C['e']] ** 2 - (pt * np.cosh(Xd[..., C['eta']])) ** 2, 0, None)
    r = np.where(rl, Xd[..., C['deltaR']], 0.0); M0 = pt.sum(1); M2 = (pt * r ** 2).sum(1)
    return dict(rl=rl, ptJ=ptJ, EJ=EJ, m2=m2, M0=M0, M2=M2, rbar=(pt * r).sum(1) / np.maximum(M0, 1e-12))


def sub(I, i): return {k: v[i] for k, v in I.items()}


def cols(pt, eta, phi, er, pr, m2, ptJ, EJ):
    out = dict(px=pt * np.cos(phi), py=pt * np.sin(phi), pz=pt * np.sinh(eta), pt=pt, ptrel=pt / ptJ[:, None], eta=eta, etarel=er,
               phi=phi, phirel=pr, deltaR=np.hypot(er, pr), costheta=np.tanh(eta))
    out['e'] = np.sqrt(m2 + (pt * np.cosh(eta)) ** 2); out['erel'] = out['e'] / EJ[:, None]
    return out


def edit16(X, I, a=None, c=None, der=0.0, rot=None, dR=None):
    """edited float32 copy on real slots.  a: angle scale (J,), c: pT scale (J,), der: Δη shift, rot: rotation angle (J,),
    dR: ΔR column only multiplied by dR (J,) (nothing else changed)"""
    Xd = X.astype(np.float64); rl = I['rl']; E = Xd.copy()
    if dR is not None:
        E[..., C['deltaR']] = np.where(rl, Xd[..., C['deltaR']] * dR[:, None], Xd[..., C['deltaR']]); return E.astype(np.float32)
    J = len(X); cc = np.ones(J) if c is None else np.asarray(c, np.float64)
    pt, eta, phi, er, pr = (Xd[..., C[k]] for k in ('pt', 'eta', 'phi', 'etarel', 'phirel'))
    er2, pr2 = er.copy(), pr.copy()
    if rot is not None:
        cs, sn = np.cos(rot)[:, None], np.sin(rot)[:, None]; er2, pr2 = cs * er - sn * pr, sn * er + cs * pr
    if a is not None: er2, pr2 = er2 * a[:, None], pr2 * a[:, None]
    er2 = er2 + der
    eta2 = eta + (er2 - er); phi2 = phi + (pr2 - pr)
    F0 = cols(pt, eta, phi, er, pr, I['m2'], I['ptJ'], I['EJ'])
    F1 = cols(pt * cc[:, None], eta2, phi2, er2, pr2, I['m2'] * cc[:, None] ** 2, I['ptJ'] * cc, I['EJ'] * cc)
    for k in F0:
        d = F1[k] - F0[k]
        if k == 'phi': d = wrap(d)
        v = Xd[..., C[k]] + d
        if k == 'phi': v = wrap(v)
        E[..., C[k]] = np.where(rl, v, Xd[..., C[k]])
    if a is not None:
        for k in ('etarot', 'phirot', 'costhetarel'): E[..., C[k]] = np.where(rl, Xd[..., C[k]] * a[:, None], Xd[..., C[k]])
    return E.astype(np.float32)


def s_of(X, I, d):
    Xd = X.astype(np.float64); pt = np.where(I['rl'], Xd[..., C['pt']], 0)
    M2T = (pt * ((Xd[..., C['etarel']] + d) ** 2 + Xd[..., C['phirel']] ** 2)).sum(1)
    M2b = (pt * (Xd[..., C['etarel']] ** 2 + Xd[..., C['phirel']] ** 2)).sum(1)
    return np.sqrt(M2T / np.maximum(M2b, 1e-30))


def dose_edits(X, I, d, seed):
    s = s_of(X, I, d)
    ang = 2 * np.arcsin(np.minimum(1.0, d / np.maximum(2 * I['rbar'], 1e-12))) * np.random.default_rng(seed).choice([-1, 1], len(X))
    E = dict(T=edit16(X, I, der=d), dilR=edit16(X, I, dR=s), dil=edit16(X, I, a=s), R=edit16(X, I, rot=ang))
    return E, dict(s_median=float(np.median(s)), half_dlogM2_mean=float(np.log(s[I['M2'] > 0]).mean()),
                   rot_median_deg=float(np.degrees(np.median(np.abs(ang)))))


def make_pairs(n_idx, npair, seed, I):
    ok = n_idx[(I['M0'][n_idx] > 0) & (I['M2'][n_idx] > 0)]
    r = np.random.default_rng(seed); b = r.choice(ok, npair); s = r.choice(ok, npair)
    while (b == s).any(): s[b == s] = r.choice(ok, int((b == s).sum()))
    return b, s


def pair_edits(X, I, b, s):
    Ib = sub(I, b); M0, M2 = I['M0'], I['M2']; out = {}
    c = M0[s] / M0[b]; out['M0'] = edit16(X[b], Ib, a=c ** -0.5, c=c)
    a2 = np.sqrt(M2[s] / M2[b]); out['M2'] = edit16(X[b], Ib, a=a2)
    cD = np.sqrt(M0[s] * M2[s] / (M0[b] * M2[b])); out['D'] = edit16(X[b], Ib, c=cD)
    meta = dict(frac_c_out_1_3=float(((c > 3) | (c < 1 / 3)).mean()), frac_a_out_1_3=float(((a2 > 3) | (a2 < 1 / 3)).mean()),
                frac_cD_out_1_3=float(((cD > 3) | (cD < 1 / 3)).mean()), sd_log_c=float(np.log(c).std()), sd_log_a=float(np.log(a2).std()))
    return out, meta


# ------------------------------------------------------------------------------------------------ network with kernels as argument
STG = ('S', 'h4', 'h5', 'z', 'logits')
CONSUMER = dict(S='d4', h4='d5', h5='d6', z='d7')


class Net:
    """exact JEDI-linear forward (as jedi_core) with the fused kernels passed as an argument, a P override, and exact / STE tails"""
    _cache = {}

    def __init__(self, jedi, KB=None, name=''):
        import jax, jax.numpy as jnp
        self.j = jedi; self.L = L = jedi.L; self.n = jedi.n; self.name = name
        KB = KB if KB is not None else jedi.KB
        self.KBn = {k: (np.asarray(v[0], np.float32), np.asarray(v[1], np.float32)) for k, v in KB.items()}
        self.KB = {k: (jnp.asarray(a_), jnp.asarray(b_)) for k, (a_, b_) in self.KBn.items()}
        key = id(jedi)
        if key not in Net._cache: Net._cache[key] = self._build()
        self.f, self.SH, self.tails, self.ste, self.feat7 = Net._cache[key]
        self.rg = {}
        for st, k in CONSUMER.items():
            q = L[k].iq.quantizer; kk, ii, ff = (np.asarray(getattr(q, a_), np.float64).ravel() for a_ in ('k', 'i', 'f'))
            w = int(np.prod(self.SH[st])); kk, ii, ff = (np.broadcast_to(v, (w,)) if v.size == 1 else v[-w:] for v in (kk, ii, ff))
            self.rg[st] = (-kk * 2.0 ** ii, 2.0 ** ii - 2.0 ** -ff)
        self.rg['logits'] = (np.full(5, -np.inf), np.full(5, np.inf))

    def _build(self):
        import jax, jax.numpy as jnp
        L = self.L

        def dense(k, x, KB, act=True, ste=False):
            q = L[k].iq(x, training=False)
            if ste: q = x + jax.lax.stop_gradient(q - x)
            z = jnp.einsum(L[k].equation, q, KB[k][0]) + KB[k][1]
            return L[k].activation(z) if act and L[k].activation is not None else z

        def pool(k, a): return jnp.sum(L[k].iq(a, training=False), axis=L[k].axes, keepdims=L[k].keepdims) * L[k].scale

        def inner(xn, KB, Pov, use):
            a = dense('l1', xn, KB); P = pool('qs', a); P = P * (1 - use) + use * Pov.reshape(P.shape)
            g = dense('dd', P, KB); s = dense('l2', a, KB)
            qs_, qd_ = L['add'].iq([s, g], training=False)
            u = dense('l3', qs_ + qd_, KB); S = pool('qs1', u)
            h4 = dense('d4', S, KB); h5 = dense('d5', h4, KB); z = dense('d6', h5, KB, act=False); lo = dense('d7', jax.nn.relu(z), KB)
            return dict(P=P, g=g, S=S, h4=h4, h5=h5, z=z, logits=lo)

        def fwd(xn, KB, Pov, use):
            r = inner(xn, KB, Pov, use); J = xn.shape[0]; return {k: v.reshape(J, -1) for k, v in r.items()}
        F = 16 if self.j.family.startswith('16') else 3
        KBs = {k: (jax.ShapeDtypeStruct(np.shape(v[0]), jnp.float32), jax.ShapeDtypeStruct(np.shape(v[1]), jnp.float32)) for k, v in self.KBn.items()}
        sh = jax.eval_shape(inner, jax.ShapeDtypeStruct((2, self.n, F), jnp.float32), KBs, jax.ShapeDtypeStruct((2, 64), jnp.float32),
                            jax.ShapeDtypeStruct((), jnp.float32))
        SH = {k: tuple(v.shape[1:]) for k, v in sh.items()}

        def tail(stage, h, KB, ste):
            J = h.shape[0]
            if stage == 'logits': return h
            x = h.reshape((J,) + SH[stage])
            if stage == 'S': x = dense('d4', x, KB, ste=ste); stage = 'h4'
            if stage == 'h4': x = dense('d5', x, KB, ste=ste); stage = 'h5'
            if stage == 'h5': x = dense('d6', x, KB, act=False, ste=ste)
            return dense('d7', jax.nn.relu(x), KB, ste=ste).reshape(J, -1)
        tails = {st: jax.jit(lambda h, KB, st=st: tail(st, h, KB, False)) for st in STG}
        ste = {st: (lambda h, KB, st=st: tail(st, h, KB, True)) for st in STG}
        feat7 = jax.jit(lambda z: L['d7'].iq(jax.nn.relu(z.reshape((z.shape[0],) + SH['z'])), training=False).reshape(z.shape[0], -1))
        return jax.jit(fwd), SH, tails, ste, feat7

    def run(self, x, keep=('logits',), Pov=None):
        out = {k: [] for k in keep}; J = len(x)
        for i in range(0, J, B):
            xb = np.asarray(x[i:i + B]); nb = len(xb)
            xn = self.j.inputs(xb, self.n)
            if nb < B: xn = np.concatenate([xn, np.zeros((B - nb,) + xn.shape[1:], np.float32)])
            if Pov is None: pv, use = np.zeros((B, 64), np.float32), np.float32(0)
            else:
                pv = np.zeros((B, 64), np.float32); pv[:nb] = Pov[i:i + B]; use = np.float32(1)
            r = self.f(xn, self.KB, pv, use)
            for k in keep: out[k].append(np.asarray(r[k][:nb], np.float32))
        return {k: np.concatenate(v) for k, v in out.items()}

    def calls(self, x): return self.run(x)['logits'].argmax(1)

    def exact(self, st, h):
        out = []
        for i in range(0, len(h), B):
            hb = np.asarray(h[i:i + B], np.float32); nb = len(hb)
            if nb < B: hb = np.concatenate([hb, np.repeat(hb[-1:], B - nb, 0)])
            out.append(np.asarray(self.tails[st](hb, self.KB))[:nb])
        return np.concatenate(out)


# ------------------------------------------------------------------------------------------------ DAS
def clip(rg, hb, h): lo, hi = rg; return np.clip(h, np.minimum(lo, hb), np.maximum(hi, hb))


def das_fit(net, st, hb, hs, lab, basec, k, seed, steps):
    import jax, jax.numpy as jnp
    n = len(lab); nv = n // 5; tr = np.arange(n - nv); va = np.arange(n - nv, n)
    chg = lab != basec; w = np.where(chg, 0.5 / max(chg[tr].sum(), 1), 0.5 / max((~chg[tr]).sum(), 1)).astype(np.float32)
    lo_, hi_ = (jnp.asarray(np.nan_to_num(v, posinf=3e38, neginf=-3e38).astype(np.float32)) for v in net.rg[st])
    ftail = net.ste[st]; KB = net.KB
    key = (id(net.j), st, k, len(tr))
    if key not in das_fit.cache:
        def loss(Wm, Hb, Hs, Y, Wt, KB, lo, hi):
            Q, _ = jnp.linalg.qr(Wm); h = jnp.clip(Hb + ((Hs - Hb) @ Q) @ Q.T, jnp.minimum(lo, Hb), jnp.maximum(hi, Hb))
            lp = jax.nn.log_softmax(ftail(h, KB)); return -(Wt * lp[jnp.arange(Y.shape[0]), Y]).sum()
        das_fit.cache[key] = jax.jit(jax.value_and_grad(loss))
    vg = das_fit.cache[key]
    Hb, Hs, Y, Wt = (jnp.asarray(a_) for a_ in (hb[tr].astype(np.float32), hs[tr].astype(np.float32), lab[tr].astype(np.int32), w[tr]))
    rng = np.random.default_rng(seed); Wm = jnp.asarray(rng.normal(size=(hb.shape[1], k)).astype(np.float32))
    m = jnp.zeros_like(Wm); v = jnp.zeros_like(Wm); b1, b2 = 0.9, 0.999; best = (-1, None, 0); chv = chg[va]; Lf = None
    for t in range(1, steps + 1):
        L_, g = vg(Wm, Hb, Hs, Y, Wt, KB, lo_, hi_); Lf = Lf if Lf is not None else float(L_)
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        Wm = Wm - 0.01 * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + 1e-8)
        if t % 50 == 0 or t == steps:
            Q = np.linalg.qr(np.asarray(Wm, np.float64))[0]
            c = eval_Q(net, st, hb[va], hs[va], Q); ok = c == lab[va]
            sc = 0.5 * ok[chv].mean() + 0.5 * ok[~chv].mean() if chv.any() and (~chv).any() else ok.mean()
            if sc > best[0]: best = (float(sc), Q, t)
    return best[1], dict(val_balanced=best[0], best_step=best[2], loss_first=Lf, loss_last=float(L_))


das_fit.cache = {}


def eval_Q(net, st, hb, hs, Q):
    return net.exact(st, clip(net.rg[st], hb, hb + ((hs - hb) @ Q) @ Q.T)).argmax(1)


def eff(c, lab, basec):
    EH = lab != basec; return float((c == lab)[EH].mean()) if EH.sum() >= 10 else None


def boot_eff(c, lab, basec, nb=1000, seed=7):
    rng = np.random.default_rng(seed); ok = (c == lab).astype(float); EH = (lab != basec).astype(float); n = len(c); v = []
    for _ in range(nb):
        i = rng.integers(0, n, n); d = EH[i].sum()
        if d >= 10: v.append((ok[i] * EH[i]).sum() / d)
    return [float(np.quantile(v, .025)), float(np.quantile(v, .975))] if v else [None, None]


def boot_diff(c1, c2, lab, basec, nb=1000, seed=8):
    rng = np.random.default_rng(seed); EH = (lab != basec).astype(float); a = (c1 == lab) * EH; b = (c2 == lab) * EH; n = len(lab); v = []
    for _ in range(nb):
        i = rng.integers(0, n, n); d = EH[i].sum()
        if d >= 10: v.append((a[i].sum() - b[i].sum()) / d)
    v = np.asarray(v)
    if not len(v): return None
    return dict(diff=float((a.sum() - b.sum()) / max(EH.sum(), 1)), ci=[float(np.quantile(v, .025)), float(np.quantile(v, .975))],
                p=float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()) + 1 / (len(v) + 1))))


def rand_Q(d, k, rng): return np.linalg.qr(rng.normal(size=(d, k)))[0]


def labels(net, PE, base_tr, base_te, reps, pairs):
    """counterfactual calls per variable (train / test pairs) and base / source calls"""
    c_all = reps['logits'].argmax(1); lab = {}
    for arm in ('train', 'test'):
        b, s = pairs[arm]; lab[arm] = dict(base=c_all[b], source=c_all[s])
        for V in VARS: lab[arm][V] = net.calls(PE[arm][V])
    return lab


def das_block(net, reps, pairs, lab, ks, nrand, steps, stages, X=None, nb=1000):
    out = {}; pv = []
    for st in stages:
        res = out.setdefault(st, {}); d = reps[st].shape[1]
        hb_tr, hs_tr = reps[st][pairs['train'][0]], reps[st][pairs['train'][1]]
        hb_te, hs_te = reps[st][pairs['test'][0]], reps[st][pairs['test'][1]]
        base_te = lab['test']['base']
        fs = lab['test']['source']           # full-stage swap at S..logits = the source's own call
        if st != 'logits':
            chk = np.abs(net.exact(st, hb_te) - reps['logits'][pairs['test'][0]]).max(); res['tail_check_max_dlogit'] = float(chk)
        for k in ks:
            if k > d: continue
            rr = np.random.default_rng(100 + k); null = {V: [] for V in VARS}
            for _ in range(nrand):
                c = eval_Q(net, st, hb_te, hs_te, rand_Q(d, k, rr))
                for V in VARS: null[V].append(eff(c, lab['test'][V], base_te))
            for V in VARS:
                Q, info = das_fit(net, st, hb_tr, hs_tr, lab['train'][V], lab['train']['base'], k, seed=10 + k, steps=steps)
                c = eval_Q(net, st, hb_te, hs_te, Q); e = eff(c, lab['test'][V], base_te)
                nv = np.array([x for x in null[V] if x is not None], np.float64)
                p = float((1 + (nv >= e).sum()) / (1 + len(nv))) if e is not None and len(nv) else None
                r = dict(train=info, eff=e, eff_ci=boot_eff(c, lab['test'][V], base_te, nb), n_eff=int((lab['test'][V] != base_te).sum()),
                         iia=float((c == lab['test'][V]).mean()), p_rot=p,
                         random=dict(mean=float(nv.mean()) if len(nv) else None, sd=float(nv.std()) if len(nv) else None, max=float(nv.max()) if len(nv) else None, n=int(len(nv)), q=np.quantile(nv, [.025, .5, .975]).tolist() if len(nv) else None),
                         full_swap=eff(fs, lab['test'][V], base_te), full_swap_ci=boot_eff(fs, lab['test'][V], base_te, nb),
                         das_minus_full_swap=boot_diff(c, fs, lab['test'][V], base_te, nb),
                         transfer={V2: eff(c, lab['test'][V2], base_te) for V2 in VARS})
                res.setdefault(V, {})[str(k)] = r; pv.append((st, V, str(k), p))
                log(f'  {net.name} {st} k{k} {V}: DAS eff {e} {r["eff_ci"]} (n {r["n_eff"]}); random {r["random"]["mean"]} q975 '
                    f'{r["random"]["q"][2] if r["random"]["q"] else None}; p_rot {p}; full swap {r["full_swap"]}; transfer {r["transfer"]}')
    ph = holm([p if p is not None else 1.0 for *_, p in pv])
    for (st, V, k, _), h in zip(pv, ph): out[st][V][k]['holm_p'] = h
    return out


def p_swap(net, X, reps, pairs, lab, nb=1000):
    """full swap at P: base jet run with P := source P; effective IIA per variable"""
    b, s = pairs['test']; c = net.run(X[b], Pov=reps['P'][s])['logits'].argmax(1); out = {}
    for V in VARS:
        out[V] = dict(eff=eff(c, lab['test'][V], lab['test']['base']), ci=boot_eff(c, lab['test'][V], lab['test']['base'], nb))
    out['calls_changed'] = float((c != lab['test']['base']).mean()); out['agree_source_call'] = float((c == lab['test']['source']).mean())
    return out


# ------------------------------------------------------------------------------------------------ ridge read-out
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


def ridge_block(reps, reps_T, reps_D, I, fit, te, sT, rng):
    tg = dict(logM0=np.log(np.maximum(I['M0'], 1e-9)), logM2=np.log(np.maximum(I['M2'], 1e-12)))
    tg['D'] = 0.5 * (tg['logM0'] + tg['logM2'])
    true_shift = dict(logM0=np.zeros(len(te)), logM2=2 * np.log(sT), D=np.log(sT))
    p = np.random.default_rng(31).permutation(fit); f1, dv = np.sort(p[:3 * len(p) // 4]), np.sort(p[3 * len(p) // 4:])
    out = {}
    for st in ('P', 'g', 'S', 'h4', 'h5', 'z', 'logits'):
        Xs = reps[st].astype(np.float64); out[st] = {}
        for v, t in tg.items():
            m = RidgeR(Xs, t, f1, dv); pr = m(Xs[te]); r2 = 1 - ((pr - t[te]) ** 2).mean() / t[te].var()
            dT = m(reps_T[st].astype(np.float64)) - pr; dD = m(reps_D[st].astype(np.float64)) - pr
            e = dict(R2_test=float(r2), alpha=float(m.alpha), shift_T=float(dT.mean()), shift_T_ci=boot_mean(dT, rng),
                     shift_dilR=float(dD.mean()), shift_dilR_ci=boot_mean(dD, rng), true_shift=float(true_shift[v].mean()))
            if abs(true_shift[v].mean()) > 0:
                e['calib_dilR'] = float(dD.mean() / true_shift[v].mean()); e['calib_dilR_ci'] = boot_ratio(dD, true_shift[v], rng)
            if abs(dD.mean()) > 1e-6:
                e['ratio_T_over_dilR'] = float(dT.mean() / dD.mean()); e['ratio_T_over_dilR_ci'] = boot_ratio(dT, dD, rng)
            out[st][v] = e
        log(f'  ridge {st}: ' + '; '.join(f'{v} R2 {e["R2_test"]:.3f} calib {e.get("calib_dilR", float("nan")):.3f} T/dilR {e.get("ratio_T_over_dilR", float("nan")):.3f}' for v, e in out[st].items()))
    return out


# ------------------------------------------------------------------------------------------------ parts
def common(nslot=64):
    X, y, idx, d = load(nslot); N = len(X); fit, te = halves(N); I = jinfo(X)
    R = dict(sample=dict(per_class=PER, seed=2521, N=N, N_fit=len(fit), N_test=len(te), excluded='sample(3000,2501), sample(1500,2502), sample(1500,2507)',
                         class_counts=np.bincount(y, minlength=5), max_col_diff=d, split='rng(2522).permutation halves'), procedure=__doc__)
    log(f'sample N {N} (fit {len(fit)}, test {len(te)}); class counts {np.bincount(y, minlength=5).tolist()}')
    return X, y, fit, te, I, R


def pairs_and_edits(X, I, fit, te):
    pairs = dict(train=make_pairs(fit, NTR, 2523, I), test=make_pairs(te, NTE, 2524, I))
    PE, meta = {}, {}
    for arm, (b, s) in pairs.items(): PE[arm], meta[arm] = pair_edits(X, I, b, s)
    return pairs, PE, meta


def part_form():
    from interp.jedi_core import Jedi
    X, y, fit, te, I, R = common(); JS = OUT / f'e25w_form{"_smoke" if SM else ""}{"_nrand%d" % args.nrand if args.nrand else ""}.json'
    def save(): JS.write_text(json.dumps(clean(R), indent=1))  # noqa: E306
    pairs, PE, R['pair_meta'] = pairs_and_edits(X, I, fit, te); log('pair edit meta', R['pair_meta'])
    sT = s_of(X[te], sub(I, te), 0.06); It = sub(I, te)
    XT, XD = edit16(X[te], It, der=0.06), edit16(X[te], It, dR=sT)
    rng = np.random.default_rng(2525); R['models'] = {}
    for name, fam in (('16f', '16-feature'), ('16f_perminv', '16-feature-perminv')):
        j = Jedi(64, fam); net = Net(j, name=name)
        keep = ('P', 'g', 'S', 'h4', 'h5', 'z', 'logits')
        reps = net.run(X, keep); ref = j.run(X[:2000])['logits']
        c0 = reps['logits'].argmax(1); M = dict(acc=float((c0 == y).mean()), acc_ci=wilson(int((c0 == y).sum()), len(y)),
                                               own_forward_vs_jedi_run=float(np.abs(reps['logits'][:2000] - ref).max()))
        log(f'{name}: accuracy {M["acc"]:.4f} {M["acc_ci"]}; own forward vs Jedi.run max |dlogit| {M["own_forward_vs_jedi_run"]:.2e}')
        lab = labels(net, PE, None, None, reps, pairs)
        M['cf_change_frac_test'] = {V: float((lab['test'][V] != lab['test']['base']).mean()) for V in VARS}
        log(f'{name}: counterfactual call != base (test pairs) {M["cf_change_frac_test"]}')
        R['models'][name] = M; save()
        M['P_full_swap'] = p_swap(net, X, reps, pairs, lab, 200 if SM else 1000); log(f'{name} P full swap {M["P_full_swap"]}')
        M['das'] = das_block(net, reps, pairs, lab, (1, 2), NRAND, STEPS, STG, nb=200 if SM else 1000); save()
        rT = net.run(XT, keep); rD = net.run(XD, keep)
        M['ridge'] = ridge_block(reps, rT, rD, I, fit, te, sT, rng); save()
        del reps, rT, rD
    import resource; R['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024; R['wall_s'] = time.time() - T0
    save(); log('done')


def call_stats(c0, ce, y, rng):
    W = y == W_; d = dict(calls_changed=float((ce != c0).mean()), acc=float((ce == y).mean()))
    m = np.zeros((5, 5), int); np.add.at(m, (c0, ce), 1); d['matrix_all'] = m
    mw = np.zeros((5, 5), int); np.add.at(mw, (c0[W], ce[W]), 1); d['matrix_W'] = mw
    d['W_new_call'] = {CL[k]: dict(f=float((ce[W] == k).mean()), ci=boot_mean((ce[W] == k).astype(float), rng)) for k in range(5)}
    fl = (ce[W] == Z_).astype(float) - (c0[W] == Z_); d['WZ_flow'] = float(fl.mean()); d['WZ_flow_ci'] = boot_mean(fl, rng)
    return d, fl


def part_dose():
    from interp.jedi_core import Jedi
    X, y, fit, te, I, R = common(); JS = OUT / f'e25w_dose{"_smoke" if SM else ""}.json'
    def save(): JS.write_text(json.dumps(clean(R), indent=1))  # noqa: E306
    rng = np.random.default_rng(2526); W = y == W_; R['models'] = {}; tests = []
    specs = [('16f_n64', 64, '16-feature'), ('16f_perminv_n64', 64, '16-feature-perminv'), ('16f_n32', 32, '16-feature'), ('16f_n128', 128, '16-feature')]
    nets = {}
    for name, n, fam in specs:
        try:
            nets[name] = Jedi(n, fam); R['models'][name] = dict(path=str(nets[name].path))
        except Exception as e:  # noqa: BLE001
            R['models'][name] = dict(absent=repr(e)); log(f'{name}: not loaded ({e!r})')
    X128 = None
    if '16f_n128' in nets:
        X128, y128, _, _ = load(128); assert np.array_equal(y128, y); I128 = jinfo(X128)
    c0s = {}
    for name, net in nets.items():
        xin = X128 if name == '16f_n128' else (X[:, :32] if name == '16f_n32' else X)
        c0 = net.run(xin, batch=B)['logits'].argmax(1); c0s[name] = c0
        R['models'][name].update(acc=float((c0 == y).mean()), acc_ci=wilson(int((c0 == y).sum()), len(y)), W_to_Z_unedited=float((c0[W] == Z_).mean()),
                                 matrix_unedited_true_vs_call=np.histogram2d(y, c0, bins=[np.arange(6) - .5] * 2)[0].astype(int), doses={})
        log(f'{name}: accuracy {R["models"][name]["acc"]:.4f}; W->Z unedited {R["models"][name]["W_to_Z_unedited"]:.3f}')
    save()
    for d in DOSES:
        E, meta = dose_edits(X, I, d, seed=int(d * 1e4)); R.setdefault('dose_meta', {})[str(d)] = meta
        E128 = dose_edits(X128, I128, d, seed=int(d * 1e4))[0] if X128 is not None else None
        for name, net in nets.items():
            Ein = E128 if name == '16f_n128' else ({k: v[:, :32] for k, v in E.items()} if name == '16f_n32' else E)
            c0 = c0s[name]; res = {}; fl = {}; zz = {}
            for arm, xe in Ein.items():
                ce = net.run(xe, batch=B)['logits'].argmax(1); res[arm], fl[arm] = call_stats(c0, ce, y, rng); zz[arm] = ce[W] == Z_
            for arm in ('T', 'dilR', 'dil'):
                res[arm]['mcnemar_Z_vs_R'] = mcnemar(zz['R'], zz[arm]); tests.append((name, str(d), arm, res[arm]['mcnemar_Z_vs_R']['p']))
            for a_, b_ in (('T', 'dilR'), ('T', 'dil'), ('dil', 'dilR')):
                if fl[b_].mean() > 0: res[f'ratio_{a_}_over_{b_}'] = dict(r=float(fl[a_].mean() / fl[b_].mean()), ci=boot_ratio(fl[a_], fl[b_], rng))
            R['models'][name]['doses'][str(d)] = res
            log(f'{name} d {d}: ' + '; '.join(f'{a} W->Z {v["W_new_call"]["Z"]["f"]:.3f} t {v["W_new_call"]["t"]["f"]:.3f} '
                                              f'q/g {v["W_new_call"]["q"]["f"] + v["W_new_call"]["g"]["f"]:.3f} W {v["W_new_call"]["W"]["f"]:.3f}'
                                              for a, v in res.items() if isinstance(v, dict) and 'W_new_call' in v) +
                f'; T/dilR {res.get("ratio_T_over_dilR", {}).get("r")}')
        del E, E128; save()
    ph = holm([t[3] for t in tests])
    for (name, d, arm, _), h in zip(tests, ph): R['models'][name]['doses'][d][arm]['holm_p_vs_R'] = h
    import resource; R['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024; R['wall_s'] = time.time() - T0
    save(); log('done')


REDRAW = ('l2', 'dd', 'l3', 'd4', 'd5', 'd6')


def variant_KB(KB, kind, s):
    rng = np.random.default_rng(1000 * {'init': 1, 'gauss': 2, 'perm': 3}[kind] + s); out = {}
    for k in sorted(KB):
        K, b = (np.asarray(v, np.float64).copy() for v in KB[k])
        if kind in ('init', 'gauss') and k in REDRAW:
            if kind == 'init':
                fi, fo = K.shape[-2], K.shape[-1]; lim = np.sqrt(6 / (fi + fo)); K = rng.uniform(-lim, lim, K.shape); b = np.zeros_like(b)
            else:
                K = rng.normal(K.mean(), K.std(), K.shape); b = rng.normal(b.mean(), b.std(), b.shape)
        elif kind == 'perm' and k != 'd7':
            for arr in (K.reshape(-1), b.reshape(-1)):
                pos = np.flatnonzero(arr != 0); arr[pos] = arr[pos][rng.permutation(len(pos))]
        out[k] = (K, b)
    return out


def refit_d7(net, z, y, fit):
    from sklearn.linear_model import LogisticRegression
    F = np.asarray(net.feat7(z.astype(np.float32)), np.float64); mu, sd = F[fit].mean(0), F[fit].std(0); live = sd > 1e-9
    K7, b7 = net.KBn['d7']; W = np.zeros((F.shape[1], 5)); b = np.zeros(5)
    if live.any() and len(np.unique(y[fit])) > 1:
        lr = LogisticRegression(C=1.0, max_iter=3000).fit((F[fit][:, live] - mu[live]) / sd[live], y[fit])
        W[live] = (lr.coef_ / sd[live]).T; b = lr.intercept_ - (mu[live] / sd[live]) @ lr.coef_.T
    KB = dict(net.KBn); KB['d7'] = (W.reshape(K7.shape).astype(np.float32), b.reshape(b7.shape).astype(np.float32))
    return KB, int(live.sum())


def part_base():
    from interp.jedi_core import Jedi
    X, y, fit, te, I, R = common(); JS = OUT / f'e25w_base{"_smoke" if SM else ""}.json'
    def save(): JS.write_text(json.dumps(clean(R), indent=1))  # noqa: E306
    rng = np.random.default_rng(2527)
    pairs, PE, R['pair_meta'] = pairs_and_edits(X, I, fit, te)
    Xt, yt, It = X[te], y[te], sub(I, te); Wt = yt == W_
    E, meta = dose_edits(Xt, It, 0.06, seed=600); R['edit_meta'] = meta
    j = Jedi(64, '16-feature'); tr = Net(j, name='trained')
    nseed = (2, 2, 2) if SM else (5, 5, 10)
    variants = [('trained', None), ('trained_refit', 'refit')] + [(f'init{s}', ('init', s)) for s in range(nseed[0])] + \
               [(f'gauss{s}', ('gauss', s)) for s in range(nseed[1])] + [(f'perm{s}', ('perm', s)) for s in range(nseed[2])]
    R['nets'] = {}
    for name, spec in variants:
        t1 = time.time()
        if spec is None: net = tr
        else:
            KB0 = tr.KBn if spec == 'refit' else variant_KB(tr.KBn, *spec)
            pre = Net(j, KB0, name=name + '_pre'); z = pre.run(X, ('z',))['z']
            KB1, nlive = refit_d7(pre, z, y, fit); net = Net(j, KB1, name=name); del pre
        reps = net.run(X, ('S', 'h4', 'h5', 'z', 'logits')); c = reps['logits'].argmax(1)
        acc_te = float((c[te] == yt).mean()); ndist = int(len(np.unique(c[te])))
        M = dict(acc_test=acc_te, acc_test_ci=wilson(int((c[te] == yt).sum()), len(yt)), acc_fit=float((c[fit] == y[fit]).mean()), n_distinct_calls=ndist,
                 call_dist_test=np.bincount(c[te], minlength=5), qualifies=bool(acc_te > 0.40 and ndist > 1))
        if spec is not None: M['d7_live_inputs'] = nlive
        c0 = c[te]; zz = {}; fl = {}; M['arms'] = {}
        for arm in ('T', 'dilR', 'R'):
            ce = net.calls(E[arm]); zz[arm] = ce[Wt] == Z_; fl[arm] = zz[arm].astype(float) - (c0[Wt] == Z_)
            M['arms'][arm] = dict(WZ=float(zz[arm].mean()), flow=float(fl[arm].mean()), flow_ci=boot_mean(fl[arm], rng), calls_changed=float((ce != c0).mean()),
                                  W_calls_changed=float((ce[Wt] != c0[Wt]).mean()))
        M['WZ_unedited'] = float((c0[Wt] == Z_).mean()); M['mcnemar_T_vs_R'] = mcnemar(zz['R'], zz['T'])
        if fl['dilR'].mean() > 0:
            M['ratio'] = float(fl['T'].mean() / fl['dilR'].mean()); M['ratio_ci'] = boot_ratio(fl['T'], fl['dilR'], rng)
        log(f'{name}: acc test {acc_te:.4f} (fit {M["acc_fit"]:.4f}); distinct calls {ndist}; qualifies {M["qualifies"]}; W->Z unedited '
            f'{M["WZ_unedited"]:.3f}; flows T {M["arms"]["T"]["flow"]:+.3f} dilR {M["arms"]["dilR"]["flow"]:+.3f} R {M["arms"]["R"]["flow"]:+.3f}; '
            f'calls changed T {M["arms"]["T"]["calls_changed"]:.3f} R {M["arms"]["R"]["calls_changed"]:.3f}; ratio {M.get("ratio")} {M.get("ratio_ci")}')
        if name != 'trained' and M['qualifies']:
            lab = labels(net, PE, None, None, reps, pairs)
            M['cf_change_frac_test'] = {V: float((lab['test'][V] != lab['test']['base']).mean()) for V in VARS}
            M['das'] = das_block(net, reps, pairs, lab, (2,), 20 if SM else 50, 60 if SM else 400, ('S', 'h4', 'h5', 'z'), nb=200 if SM else 500)
        elif name == 'trained':
            lab = labels(net, PE, None, None, reps, pairs)
            M['das'] = das_block(net, reps, pairs, lab, (2,), 20 if SM else 50, 60 if SM else 400, ('S', 'h4', 'h5', 'z'), nb=200 if SM else 500)
        M['wall_s'] = time.time() - t1; R['nets'][name] = M; save(); del reps
    trd = R['nets']['trained']; rt = trd.get('ratio'); ft = trd['arms']['T']['flow']
    q = [n for n, m in R['nets'].items() if n not in ('trained', 'trained_refit') and m['qualifies']]
    def dist(n): r = R['nets'][n].get('ratio'); return abs(r - 1) if r is not None else np.inf  # noqa: E306
    n = len(q)
    R['test'] = dict(qualifying=q, n_qualifying=n, n_baselines=len(variants) - 2,
                     p_ratio=float((1 + sum(dist(b) <= abs(rt - 1) for b in q)) / (1 + n)) if rt is not None else None,
                     p_flow=float((1 + sum(R['nets'][b]['arms']['T']['flow'] >= ft for b in q)) / (1 + n)),
                     baseline_ratios={b: R['nets'][b].get('ratio') for b in q}, baseline_flows_T={b: R['nets'][b]['arms']['T']['flow'] for b in q},
                     trained_ratio=rt, trained_flow_T=ft, refit_ratio=R['nets']['trained_refit'].get('ratio'))
    log('test', R['test'])
    import resource; R['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024; R['wall_s'] = time.time() - T0
    save(); log('done')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    dict(form=part_form, dose=part_dose, base=part_base)[args.part]()
