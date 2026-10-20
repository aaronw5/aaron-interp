"""X13 (claims A4 / A10m, exploratory; reviewer question): what sets the size of the response to a rigid translation of the
jet constituents (the stored-axis "mass" response) across networks: the radial route (per-particle distances r_i from the
stored axis) vs pairwise / relative structure, and architecture vs training seed.

Networks (all existing; no training):
  jedi   official JEDI-linear 3-feature n32, n64, n128 and 3-feature-perminv n64 (interp.jedi_core.Jedi); the 8 X12 retrained
         JEDI-linear n64 copies (hgq_s0-2, hgq-r150_s0-2 = HGQ; float_s0-1 = quantizers off; interp.x12_retrained_jedi.Net).
  flow   X9 PFN / EFN (interp.x9_efn_pfn.Flow, exact numpy forward; runs in the jax env, no energyflow import needed).
  s64 / s32  X0/X1 torch small models (train/models.py): Deep Sets, JEDI-style Deep Sets, Linformer, transformers, N 64 / N 32.
  Untrained references: 3 redraws per architecture (JEDI: jetdistill.network.redraw_untrained seeds 0-2 = the project's
  untrained control; torch: default PyTorch init under torch.manual_seed(5000 + k); PFN / EFN: he_uniform kernels, zero biases,
  rng 1313_00 + k).  For untrained networks the native output is often constant, so a multinomial logistic head is refit on the
  last hidden layer (fit half) = the "random body + refit read-out" reference for call-based numbers.
Sample: interp.jedi_core.sample(per_class=1000, seed=1313), N = 5,000 test jets, identical jets for every network.  n = 128
  networks get the same rows of val_ptetaphi_f16.npy (first 64 particles checked equal).  A network with n inputs sees the first
  n particles; every edit is computed on those n particles (moment-matched edits use the n-particle moments).
  Fixed split rng(1313): half "fit" (read-out / refit head), half "test" (read-out responses).
Edits (real particles only; padding untouched), c = 0.06 rad:
  T_k   rigid translation by c at azimuth 2 pi k / 8, k = 0..7 (mirror pairs k, k + 4 give the even / odd split).  Invariant
        mass unchanged exactly (boost along z + rotation in phi).
  R_k   radial-only: each particle keeps its azimuth theta_i about the stored axis; r_i -> |p_i + c_k| (the same per-particle
        r change as T_k); pairwise distances change, M2 changes as under T_k.
  dil   matched dilation: angles x s per jet, log s = mean_k 1/2 Delta log M2(T_k), so 1/2 Delta log(M0 M2) equals its
        azimuth mean under T for every jet.
  A_d   pair route (E24 A'): per-particle azimuth scramble at fixed r_i, then translation + rescale about the centroid so that
        M0, M1 (vector) and M2 about the stored axis are held exactly; 3 draws (rng 131310 + d).  Changes pairwise distances.
  rot   null: rigid rotation about the stored axis by a random angle per jet (rng 131320); r_i, pairwise distances, m, M0, M2
        unchanged.
Read-outs per network: (i) ridge read-out of log m (invariant mass of the n particles) on the last hidden layer (input of the
  output layer; standardised, alpha chosen on the last quarter of the fit half, refit on the fit half; E6v / X2 procedure),
  fitted on unedited fit-half jets, applied to test-half jets before / after each edit; (ii) the same on the pooled latent
  (JEDI S, Deep Sets sum, PFN / EFN L; transformers: the pooled vector = the last hidden layer); (iii) native calls.
Stored per network (npz): calls (edits x J), read-out predictions on the test half, refit-head calls (untrained).
Analysis (--analyze; jet bootstrap B = 1000 with the same jet indices for every network):
  R      = mean_test[ mean_k d(T_k) ] / mean_test[ d(dil) ]                     (read-out; primary)
  R_fl   = W->Z flow(T, azimuth mean) / W->Z flow(dil)                          (calls)
  Q_rad  = mean d(R_k) / mean d(T_k)  (read-out and flow);  R_rad = mean d(R_k) / mean d(dil)
  Q_pair = mean |d(A)| / mean |d(T_k)| (read-out) and calls changed(A) / calls changed(T)
  odd    = first azimuthal harmonic amplitude of d(T_k) per jet, mean / |mean even|
  Across trained networks: Spearman(R, Q_rad) with a bootstrap over networks and jets; partial Spearman given accuracy and
  read-out R^2; Spearman(R, accuracy), (R, R^2); variance decomposition of log R into between-family (same architecture and
  recipe, different seeds) and within-family parts, permutation of family labels (10,000); paired seed differences with jet
  bootstrap CIs.  Per network: d(T) vs d(rot) and Q_rad = 1 (paired bootstrap p), Holm over networks.
Run:
  ./job.sh x13-run-jedi 'python -m interp.x13_axis_size --group jedi'
  ./job.sh x13-run-flow 'python -m interp.x13_axis_size --group flow'
  AI_PY=conda ./job.sh x13-run-s64 'python -m interp.x13_axis_size --group s64'   (s32 likewise)
  ./job.sh x13-ana 'python -m interp.x13_axis_size --analyze'  -> $AI_RESULTS/jedi/x13_summary.json, x13_axis_size.png
"""
import argparse, gc, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
NETS = OUT / 'x13_nets'
SMALL = Path(os.environ.get('AI_SMALL_MODELS', '/j-jepa-vol/aaron-interp/models/small'))
C = 0.06; AZ = np.arange(8) * np.pi / 4; NA = 3; PER, SEED = 1000, 1313
W_, Z_ = 2, 3; B = 1000; T0 = time.time()
EDITS = ['0'] + [f'T{k}' for k in range(8)] + [f'R{k}' for k in range(8)] + ['dil'] + [f'A{d}' for d in range(NA)] + ['rot']
ALPHAS = 10.0 ** np.arange(-4, 4.1, 0.5)
OFFICIAL = [('jedi-3f-n32', 32, '3-feature'), ('jedi-3f-n64', 64, '3-feature'), ('jedi-3f-n128', 128, '3-feature'),
            ('jedi-perminv-n64', 64, '3-feature-perminv')]
X12 = ['hgq_s0', 'hgq_s1', 'hgq_s2', 'hgq-r150_s0', 'hgq-r150_s1', 'hgq-r150_s2', 'float_s0', 'float_s1']
FLOWS = ['pfn_64-64-32_s0', 'pfn_64-64-32_s1', 'efn_100-100-128_s0', 'efn_100-100-128_s1', 'pfn_100-100-128_s0',
         'pfn_100-100-128_s1', 'efn_100-100-64_s0', 'efn_64-64-32_s0', 'pfn_100-100-64_s0']
S64 = ['dsets-d64-n64', 'jedis-d64-n64', 'linf-1b-d32-k4-n64', 'linf-2b-d32-k8-n64', 'tf-1b-d16-n64', 'tf-1b-d32-n64',
       'tf-2b-d16-n64', 'tf-2b-d32-n64', 'tf-2b-d32-n64-e100', 'tf-2b-d64-n64-e100']
S32 = ['dsets-d64-n32', 'linf-1b-d32-k4-n32', 'linf-2b-d32-k8-n32', 'tf-1b-d16-n32', 'tf-1b-d32-n32', 'tf-2b-d16-n32',
       'tf-2b-d32-n32']


def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.ndarray): return clean(o.tolist())
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    return o


# ================================================================================================ sample and edits
def load_sample():
    from interp.jedi_core import sample
    from jetdistill.config import RESULTS, DATA
    x64, y, _ = sample(per_class=PER, seed=SEED)
    yy = np.asarray(np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')['y'])
    rng = np.random.default_rng(SEED)
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(yy == c), PER, replace=False) for c in range(5)]))
    assert np.array_equal(yy[idx], y)
    V = np.load(Path(DATA) / 'val_ptetaphi_f16.npy', mmap_mode='r')
    x128 = np.asarray(V[idx, :128]).astype(np.float32)
    assert np.array_equal(x128[:, :64], np.asarray(x64, np.float32)), 'val_ptetaphi rows != full_test rows'
    p = np.random.default_rng(SEED).permutation(len(y)); fit, te = np.sort(p[:len(y) // 2]), np.sort(p[len(y) // 2:])
    return x128, y, fit, te


def moments(x):
    x = np.asarray(x, np.float64); pt, eta, phi = x[..., 0], x[..., 1], x[..., 2]
    M0 = pt.sum(1); M1 = np.stack([(pt * eta).sum(1), (pt * phi).sum(1)], -1); M2 = (pt * (eta ** 2 + phi ** 2)).sum(1)
    p4 = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)], -1).sum(1)
    m = np.sqrt(np.maximum(p4[:, 3] ** 2 - (p4[:, :3] ** 2).sum(1), 0.0))
    return dict(M0=M0, M1=M1, M2=M2, m=m)


def polar(x):
    x = np.asarray(x, np.float64); return np.hypot(x[..., 1], x[..., 2]), np.arctan2(x[..., 2], x[..., 1]), x[..., 0] > 0


def from_polar(x, r, th, real):
    e = np.array(x, np.float64, copy=True)
    e[..., 1] = np.where(real, r * np.cos(th), x[..., 1]); e[..., 2] = np.where(real, r * np.sin(th), x[..., 2]); return e


def hold_moments(x_new, x_ref):
    """E24 A': translate + rescale the real particles of x_new about their centroid so that M1 (vector) and M2 equal x_ref's"""
    a, b = moments(x_ref), moments(x_new); real = np.asarray(x_new)[..., 0] > 0
    c = a['M1'] / a['M0'][:, None]; c2 = b['M1'] / b['M0'][:, None]
    e = np.array(x_new, np.float64, copy=True); p = e[..., 1:3]
    num = a['M2'] - a['M0'] * (c ** 2).sum(1); den = b['M2'] - b['M0'] * (c2 ** 2).sum(1)
    s = np.sqrt(np.maximum(num, 0) / np.maximum(den, 1e-300))
    q = c[:, None, :] + s[:, None, None] * (p - c2[:, None, :])
    e[..., 1:3] = np.where(real[..., None], q, p); return e


def make_edits(x):
    """x (J, n, 3) -> {edit: float32 array}; all real-particle-only"""
    x = np.asarray(x, np.float32); real = x[..., 0] > 0; r, th, _ = polar(x); M = moments(x); E = {'0': x}
    for k, a in enumerate(AZ):
        e = x.copy(); e[..., 1] += np.float32(C * np.cos(a)) * real; e[..., 2] += np.float32(C * np.sin(a)) * real; E[f'T{k}'] = e
        r2 = np.hypot(x[..., 1].astype(np.float64) + C * np.cos(a), x[..., 2].astype(np.float64) + C * np.sin(a))
        E[f'R{k}'] = from_polar(x, r2, th, real).astype(np.float32)
    M2k = np.stack([M['M2'] + 2 * C * (M['M1'][:, 0] * np.cos(a) + M['M1'][:, 1] * np.sin(a)) + M['M0'] * C ** 2 for a in AZ])
    s = np.exp(0.5 * np.log(M2k / np.maximum(M['M2'], 1e-30)).mean(0))     # 1/2 dlog(M0 M2) = its azimuth mean under T_k, per jet
    e = x.copy(); e[..., 1:3] = np.where(real[..., None], x[..., 1:3] * s[:, None, None].astype(np.float32), x[..., 1:3]); E['dil'] = e
    for d in range(NA):
        t = np.random.default_rng(131310 + d).uniform(0, 2 * np.pi, th.shape)
        E[f'A{d}'] = hold_moments(from_polar(x, r, t, real), x).astype(np.float32)
    g = np.random.default_rng(131320).uniform(0, 2 * np.pi, len(x))
    E['rot'] = from_polar(x, r, th + g[:, None], real).astype(np.float32)
    return E


def edit_checks(E):
    x = E['0']; M = moments(x); r0, _, real = polar(x); out = {}; sub = slice(0, 400)
    def pdist(z):
        z = np.asarray(z[sub], np.float64); d = np.hypot(z[:, :, None, 1] - z[:, None, :, 1], z[:, :, None, 2] - z[:, None, :, 2])
        rr = real[sub]; return d, rr[:, :, None] & rr[:, None, :]
    d0, pm = pdist(x)
    for k, e in E.items():
        Me = moments(e); r, _, _ = polar(e); dk, _ = pdist(e)
        out[k] = dict(dhalflogM0M2=float(np.mean(0.5 * np.log(Me['M0'] * Me['M2'] / (M['M0'] * M['M2'])))),
                      dlogm=float(np.mean(np.log(np.maximum(Me['m'], 1e-6) / np.maximum(M['m'], 1e-6)))),
                      abs_dlogm=float(np.mean(np.abs(np.log(np.maximum(Me['m'], 1e-6) / np.maximum(M['m'], 1e-6))))),
                      mean_abs_dr_mrad=float(1e3 * np.abs(r - r0)[real].mean()),
                      mean_abs_dpair_mrad=float(1e3 * np.abs(dk - d0)[pm].mean()),
                      max_rel_dM2=float(np.max(np.abs(Me['M2'] / M['M2'] - 1))),
                      dM1_over_M0_mrad=float(1e3 * np.median(np.linalg.norm(Me['M1'] - M['M1'], axis=1) / M['M0'])),
                      padding_changed=int((np.asarray(e)[~real] != np.asarray(x)[~real]).sum()))
    # radial edit reproduces the translation's per-particle r exactly
    out['R_vs_T_max_abs_dr'] = float(max(np.abs(polar(E[f'R{k}'])[0] - polar(E[f'T{k}'])[0])[real].max() for k in range(8)))
    return out


# ================================================================================================ read-outs
class Ridge:
    """E6v / X2 read-out: standardised inputs, alpha chosen on the last quarter of the fit half, refit on the fit half"""
    def __init__(self, X, t):
        n = len(X); a_, b_ = np.arange(n * 3 // 4), np.arange(n * 3 // 4, n)
        self.mu = X.mean(0); self.sd = X.std(0); self.sd[self.sd < 1e-9] = np.inf
        Z = (X - self.mu) / self.sd; best = None
        for al in ALPHAS:
            w, b0 = self._fit(Z[a_], t[a_], al); e = ((Z[b_] @ w + b0 - t[b_]) ** 2).mean()
            if best is None or e < best[0]: best = (e, al)
        self.alpha = float(best[1]); self.w, self.b0 = self._fit(Z, t, self.alpha)

    @staticmethod
    def _fit(Z, t, a):
        zm, tm = Z.mean(0), t.mean(); Zc = Z - zm
        w = np.linalg.solve(Zc.T @ Zc + a * len(Z) * np.eye(Z.shape[1]), Zc.T @ (t - tm)); return w, tm - zm @ w

    def __call__(self, X): return ((X - self.mu) / self.sd) @ self.w + self.b0


def measure(name, fwd, x128, y, fit, te, n, meta, cache, refit=False):
    """fwd(x (J, n, 3) float32) -> dict(logits, last, pool).  Saves NETS/<name>.npz and returns a summary row."""
    if cache.get('n') != n:
        cache.clear(); cache['n'] = n; cache['E'] = make_edits(x128[:, :n]); cache['chk'] = edit_checks(cache['E'])
        cache['logm'] = np.log(np.maximum(moments(x128[:, :n])['m'], 1e-3))
        log(f'edits for n={n}: ' + '; '.join(f'{k}: ½Δlog(M0M2) {v["dhalflogM0M2"]:+.4f} Δlog m {v["dlogm"]:+.4f} |Δr| {v["mean_abs_dr_mrad"]:.1f} '
                                            f'|Δpair| {v["mean_abs_dpair_mrad"]:.1f} mrad' for k, v in cache['chk'].items() if k in ('T0', 'R0', 'dil', 'A0', 'rot'))
            + f'; R vs T max |Δr| {cache["chk"]["R_vs_T_max_abs_dr"]:.1e}')
    E, logm = cache['E'], cache['logm']
    t1 = time.time(); R0 = fwd(E['0']); c0 = R0['logits'].argmax(1)
    acc = float((c0 == y).mean()); const = float(np.bincount(c0, minlength=5).max() / len(c0))
    rd = {k: Ridge(R0[k][fit], logm[fit]) for k in ('last', 'pool')}
    r2 = {k: float(1 - ((rd[k](R0[k][te]) - logm[te]) ** 2).sum() / ((logm[te] - logm[te].mean()) ** 2).sum()) for k in rd}
    head = None
    if refit:
        from sklearn.linear_model import LogisticRegression
        X = R0['last']; mu, sd = X[fit].mean(0), X[fit].std(0); ok = sd > 1e-9
        if ok.sum() >= 1:
            head = (LogisticRegression(max_iter=1000, C=1.0).fit((X[fit][:, ok] - mu[ok]) / sd[ok], y[fit]), mu, sd, ok)
    calls = np.zeros((len(EDITS), len(y)), np.int8); pred = {k: np.zeros((len(EDITS), len(te)), np.float32) for k in rd}
    rcalls = np.zeros((len(EDITS), len(te)), np.int8) if head else None
    for i, e in enumerate(EDITS):
        Re = R0 if e == '0' else fwd(E[e]); calls[i] = Re['logits'].argmax(1)
        for k in rd: pred[k][i] = rd[k](Re[k][te])
        if head:
            h, mu, sd, ok = head; rcalls[i] = h.predict((Re['last'][te][:, ok] - mu[ok]) / sd[ok])
    NETS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(NETS / f'{name}.npz', calls=calls, pred_last=pred['last'], pred_pool=pred['pool'],
                        **({'rcalls': rcalls} if head else {}))
    W = y == W_
    fl = lambda cc: float((cc[W] == Z_).mean() - (c0[W] == Z_).mean())
    fT = np.mean([fl(calls[EDITS.index(f'T{k}')]) for k in range(8)]); fD = fl(calls[EDITS.index('dil')])
    fR = np.mean([fl(calls[EDITS.index(f'R{k}')]) for k in range(8)])
    dT = np.mean([pred['last'][EDITS.index(f'T{k}')] - pred['last'][0] for k in range(8)], 0)
    dR = np.mean([pred['last'][EDITS.index(f'R{k}')] - pred['last'][0] for k in range(8)], 0)
    dD = pred['last'][EDITS.index('dil')] - pred['last'][0]
    row = dict(name=name, n=n, **meta, acc=acc, max_call_share=const, readout_r2=r2, readout_alpha={k: v.alpha for k, v in rd.items()},
               quick=dict(R=float(dT.mean() / dD.mean()) if abs(dD.mean()) > 1e-9 else None,
                          Q_rad=float(dR.mean() / dT.mean()) if abs(dT.mean()) > 1e-9 else None,
                          flow_T=float(fT), flow_dil=fD, flow_R=float(fR), W_to_Z_0=float((c0[W] == Z_).mean())),
               refit_head=bool(head), seconds=time.time() - t1)
    log(f'{name:28s} n{n:<3d} acc {acc:.4f} (max call share {const:.2f}) R² last {r2["last"]:.3f} pool {r2["pool"]:.3f} | '
        f'read-out ΔT {dT.mean():+.4f} Δdil {dD.mean():+.4f} ΔR {dR.mean():+.4f} → R {row["quick"]["R"]} Q_rad {row["quick"]["Q_rad"]} | '
        f'flow T {fT:+.3f} dil {fD:+.3f} R {fR:+.3f} [{row["seconds"]:.0f}s]')
    return row


# ================================================================================================ networks
def jedi_fwd(net):
    def f(x):
        r = net.run(x, keep=('z', 'S', 'logits')); return dict(logits=r['logits'], last=np.maximum(r['z'], 0), pool=r['S'])
    return f


def x12_net(m, useed=None):
    from interp.x12_retrained_jedi import Net, model_path
    mp, _, tp = model_path(m, 'final'); net = Net('x12', mp)
    if useed is not None:
        from jetdistill.network import redraw_untrained
        redraw_untrained(net.model, useed)
        for k in ('l1', 'l2', 'dd', 'l3', 'd4', 'd5', 'd6', 'd7'):
            K, b = net.L[k].get_fused_qkernel_and_qbias(False, net.L[k].moving_mean, net.L[k].moving_variance)
            net.KB[k] = (np.asarray(K, np.float64), np.asarray(b, np.float64))
        net.build()
    acc = json.loads(Path(tp).read_text()).get('accuracy') if Path(tp).exists() else None
    return net, acc


def free():
    gc.collect()
    try:
        import jax, keras; jax.clear_caches(); keras.backend.clear_session()
    except Exception: pass


def run_jedi(x128, y, fit, te, rows, cache):
    from interp.jedi_core import Jedi
    from jetdistill.config import DATA
    oa = json.loads((Path(DATA) / 'official_acc.json').read_text()) if (Path(DATA) / 'official_acc.json').exists() else {}
    for nm, n, fam in sorted(OFFICIAL, key=lambda t: t[1]):
        for us in (None, 0, 1, 2):
            j = Jedi(n, fam, untrained=us is not None, seed=us or 0)
            tag = nm if us is None else f'{nm}~u{us}'
            rows.append(measure(tag, jedi_fwd(j), x128, y, fit, te, n, dict(family=nm, cls='jedi-linear', trained=us is None,
                                seed=0 if us is None else us, ref_acc=oa.get(fam, {}).get(str(n)) if isinstance(oa.get(fam), dict) else None),
                                cache, refit=us is not None))
            del j; free()
    for m in X12 + ['hgq_s0~u0', 'hgq_s0~u1', 'hgq_s0~u2', 'float_s0~u0', 'float_s0~u1', 'float_s0~u2']:
        base, us = (m.split('~u')[0], int(m.split('~u')[1])) if '~u' in m else (m, None)
        net, acc = x12_net(base, us); arm = base.split('_s')[0]
        rows.append(measure(f'x12-{m}', jedi_fwd(net), x128, y, fit, te, 64,
                            dict(family=f'x12-{arm}' if us is None else f'x12-{arm}', cls='jedi-linear' if arm != 'float' else 'jedi-linear-float',
                                 trained=us is None, seed=int(base.split('_s')[1]) if us is None else us, ref_acc=acc), cache, refit=us is not None))
        del net; free()


def run_flow(x128, y, fit, te, rows, cache):
    from interp.x9_efn_pfn import Flow
    import copy
    for nm in FLOWS:
        F = Flow(nm); arch = nm.rsplit('_s', 1)[0]
        f = lambda x, F=F: (lambda r: dict(logits=r['logits'], last=r['h'], pool=r['L']))(F.run(x, keep=('L', 'h', 'logits')))
        rows.append(measure(nm, f, x128, y, fit, te, 64, dict(family=arch, cls=nm.split('_')[0], trained=True, seed=int(nm.rsplit('_s', 1)[1]),
                                                               ref_acc=F.info.get('acc_test')), cache))
        if nm.endswith('_s0'):
            for k in range(3):
                r = np.random.default_rng(131300 + k); U = copy.copy(F)
                hu = lambda W: r.uniform(-1, 1, W.shape) * np.sqrt(6.0 / W.shape[0])
                U.phi = [(hu(W), np.zeros_like(b)) for W, b in F.phi]; U.F = [(hu(W), np.zeros_like(b)) for W, b in F.F]
                U.out = (hu(F.out[0]), np.zeros_like(F.out[1]))
                fu = lambda x, U=U: (lambda r_: dict(logits=r_['logits'], last=r_['h'], pool=r_['L']))(U.run(x, keep=('L', 'h', 'logits')))
                rows.append(measure(f'{arch}~u{k}', fu, x128, y, fit, te, 64, dict(family=arch, cls=nm.split('_')[0], trained=False, seed=k,
                                                                                   ref_acc=None), cache, refit=True))


def torch_runner(net, arch):
    import torch
    sh = np.asarray(arch['inputs']['shift'], np.float32); sc = np.asarray(arch['inputs']['scale'], np.float32); cap = {}
    net.head.register_forward_pre_hook(lambda m, inp: cap.__setitem__('last', inp[0]))
    if hasattr(net, 'rho'): net.rho.register_forward_pre_hook(lambda m, inp: cap.__setitem__('pool', inp[0]))

    def f(x, batch=1000):
        out = dict(logits=[], last=[], pool=[])
        with torch.no_grad():
            for i in range(0, len(x), batch):
                xb = np.asarray(x[i:i + batch], np.float32); m = xb[..., 0] > 0; xn = ((xb - sh) / sc) * m[..., None]
                lo = net(torch.from_numpy(xn.astype(np.float32)), torch.from_numpy(m))
                out['logits'].append(lo.double().numpy()); out['last'].append(cap['last'].double().numpy())
                out['pool'].append(cap.get('pool', cap['last']).double().numpy())
        return {k: np.concatenate(v) for k, v in out.items()}
    return f


def run_small(models, x128, y, fit, te, rows, cache):
    import importlib.util, torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
    spec = importlib.util.spec_from_file_location('small_models', ROOT / 'train' / 'models.py')
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    for run in models:
        seeds = sorted(int(p.name[4:]) for p in (SMALL / run).glob('seed*') if (p / 'best.pt').exists())
        for s in seeds + ['u0', 'u1', 'u2']:
            d = SMALL / run / f'seed{seeds[0] if isinstance(s, str) else s}'; arch = json.loads((d / 'arch.json').read_text())
            assert not arch['cls_token'] and not arch['inputs'].get('recentre') and not arch['inputs'].get('aug')
            if isinstance(s, str): torch.manual_seed(5000 + int(s[1]))
            net = M.build_model(arch['spec'], arch['n_particles'], cls_token=False, final_ln=arch['final_ln'])
            if not isinstance(s, str): net.load_state_dict(torch.load(d / 'best.pt', map_location='cpu'))
            net.eval(); n = arch['n_particles']
            ta = json.loads((d / 'test.json').read_text()).get('accuracy') if not isinstance(s, str) else None
            nm = f'{run}_s{s}' if not isinstance(s, str) else f'{run}~{s}'
            rows.append(measure(nm, torch_runner(net, arch), x128, y, fit, te, n,
                                dict(family=run, cls=arch['spec']['arch'], trained=not isinstance(s, str),
                                     seed=s if not isinstance(s, str) else int(s[1]), ref_acc=ta), cache, refit=isinstance(s, str)))


def run_group(a):
    x128, y, fit, te = load_sample(); log(f'sample(per_class={PER}, seed={SEED}): N {len(y)}, fit {len(fit)}, test {len(te)}; '
                                          f'real particles/jet (128 list) {(x128[..., 0] > 0).sum(1).mean():.1f}, >64: {((x128[..., 0] > 0).sum(1) > 64).mean():.3f}')
    rows, cache = [], {}
    fo = OUT / f'x13_{a.group}.json'
    try:
        if a.group == 'jedi': run_jedi(x128, y, fit, te, rows, cache)
        elif a.group == 'flow': run_flow(x128, y, fit, te, rows, cache)
        elif a.group in ('s64', 's32'): run_small(S64 if a.group == 's64' else S32, x128, y, fit, te, rows, cache)
        else: raise ValueError(a.group)
    finally:
        fo.write_text(json.dumps(clean(dict(group=a.group, sample=dict(per_class=PER, seed=SEED, N=len(y)), c=C, edits=EDITS,
                                            edit_checks={str(cache.get('n')): cache.get('chk')}, rows=rows)), indent=1))
        log('wrote', fo, len(rows), 'networks')


# ================================================================================================ analysis
def holm(ps):
    ps = np.asarray(ps, np.float64); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o): run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def ci(v):
    v = np.asarray(v, np.float64); v = v[np.isfinite(v)]
    return [float(np.quantile(v, .025)), float(np.quantile(v, .975))] if len(v) else [None, None]


def bp(v, ref=0.0):
    v = np.asarray(v, np.float64); v = v[np.isfinite(v)]
    if not len(v): return None
    return float(min(1.0, 2 * (1 + min((v <= ref).sum(), (v >= ref).sum())) / (len(v) + 1)))


def rank(v): from scipy.stats import rankdata; return rankdata(v)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4: return np.nan
    ra, rb = rank(a[ok]), rank(b[ok]); return float(np.corrcoef(ra, rb)[0, 1])


def partial_spearman(a, b, Z):
    a, b = np.asarray(a, float), np.asarray(b, float); Z = np.asarray(Z, float).reshape(len(a), -1)
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(Z).all(1)
    if ok.sum() < Z.shape[1] + 4: return np.nan
    X = np.c_[np.ones(ok.sum()), np.stack([rank(z) for z in Z[ok].T], 1)]
    res = lambda v: v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
    return float(np.corrcoef(res(rank(a[ok])), res(rank(b[ok])))[0, 1])


def net_stats(name, y, te, I_te, I_all):
    """per-network effects and B jet-bootstrap replicates (same indices for every network)"""
    d = np.load(NETS / f'{name}.npz'); ix = {e: i for i, e in enumerate(EDITS)}; out = {}; reps = {}
    for lay in ('last', 'pool'):
        P = d[f'pred_{lay}'].astype(np.float64); D = P - P[0]
        dT = D[[ix[f'T{k}'] for k in range(8)]]; dR = D[[ix[f'R{k}'] for k in range(8)]]
        Te, Re, Di = dT.mean(0), dR.mean(0), D[ix['dil']]; Ab = np.abs(D[[ix[f'A{k}'] for k in range(NA)]]).mean(0)
        As = D[[ix[f'A{k}'] for k in range(NA)]].mean(0); TabsK = np.abs(dT).mean(0); Ro = D[ix['rot']]; Rabs = np.abs(Ro)
        a1 = (dT * np.cos(AZ)[:, None]).sum(0) / 4; b1 = (dT * np.sin(AZ)[:, None]).sum(0) / 4; odd = np.hypot(a1, b1)
        V = dict(T=Te, R=Re, D=Di, Aabs=Ab, A=As, Tabs=TabsK, rot=Ro, rotabs=Rabs, odd=odd, TmR=Te - Re, Tmrot=Te - Ro)
        M = {k: v.mean() for k, v in V.items()}; Bm = {k: v[I_te].mean(1) for k, v in V.items()}
        f = dict(R=('T', 'D'), Q_rad=('R', 'T'), R_rad=('R', 'D'), Q_pair=('Aabs', 'Tabs'), Q_pair_signed=('A', 'T'), R_pair=('A', 'D'),
                 rot_over_dil=('rot', 'D'), rotabs_over_Tabs=('rotabs', 'Tabs'), odd_over_even=('odd', 'T'),
                 Rpair_abs=('Aabs', 'D'), Rrot_abs=('rotabs', 'D'))
        e = {k: float(M[a] / M[b]) if abs(M[b]) > 1e-12 else None for k, (a, b) in f.items()}
        eb = {k: Bm[a] / np.where(np.abs(Bm[b]) > 1e-12, Bm[b], np.nan) for k, (a, b) in f.items()}
        out[lay] = dict(mean={k: float(v) for k, v in M.items()}, **e, ci={k: ci(v) for k, v in eb.items()},
                        p_T_vs_rot=bp(Bm['Tmrot']), p_Qrad_eq1=bp(Bm['TmR']), p_T_ne0=bp(Bm['T']))
        reps[lay] = eb
    # calls (native) on all jets; W -> Z flow over true W
    c = d['calls'].astype(int); W = y == W_; iW = np.flatnonzero(W)
    z = (c[:, W] == Z_).astype(np.float64); fz = z - z[0]
    ch = (c != c[0]).astype(np.float64)
    FV = dict(T=fz[[ix[f'T{k}'] for k in range(8)]].mean(0), R=fz[[ix[f'R{k}'] for k in range(8)]].mean(0), D=fz[ix['dil']],
              A=fz[[ix[f'A{k}'] for k in range(NA)]].mean(0), rot=fz[ix['rot']])
    CV = dict(T=ch[[ix[f'T{k}'] for k in range(8)]].mean(0), R=ch[[ix[f'R{k}'] for k in range(8)]].mean(0), D=ch[ix['dil']],
              A=ch[[ix[f'A{k}'] for k in range(NA)]].mean(0), rot=ch[ix['rot']])
    IW = np.random.default_rng(131331).integers(0, len(iW), (B, len(iW)))
    FB = {k: v[IW].mean(1) for k, v in FV.items()}; CB = {k: v[I_all].mean(1) for k, v in CV.items()}
    sdiv = lambda a, b: float(a / b) if abs(b) > 1e-12 else None
    out['calls'] = dict(flow={k: float(v.mean()) for k, v in FV.items()}, flow_ci={k: ci(v) for k, v in FB.items()},
                        changed={k: float(v.mean()) for k, v in CV.items()}, changed_ci={k: ci(v) for k, v in CB.items()},
                        R_fl=sdiv(FV['T'].mean(), FV['D'].mean()), R_fl_ci=ci(FB['T'] / np.where(np.abs(FB['D']) > 1e-12, FB['D'], np.nan)),
                        Q_rad_fl=sdiv(FV['R'].mean(), FV['T'].mean()), Q_rad_fl_ci=ci(FB['R'] / np.where(np.abs(FB['T']) > 1e-12, FB['T'], np.nan)),
                        Q_pair_cc=sdiv(CV['A'].mean(), CV['T'].mean()), Q_pair_cc_ci=ci(CB['A'] / np.where(CB['T'] > 0, CB['T'], np.nan)),
                        rot_cc_over_T=sdiv(CV['rot'].mean(), CV['T'].mean()),
                        p_flowT_vs_rot=bp(FB['T'] - FB['rot']))
    reps['calls'] = dict(R_fl=FB['T'] / np.where(np.abs(FB['D']) > 1e-12, FB['D'], np.nan),
                         Q_rad_fl=FB['R'] / np.where(np.abs(FB['T']) > 1e-12, FB['T'], np.nan))
    if 'rcalls' in d.files:       # refit head on the test half
        rc = d['rcalls'].astype(int); Wt = y[te] == W_; zr = (rc[:, Wt] == Z_).astype(np.float64); fr = zr - zr[0]
        fT = fr[[ix[f'T{k}'] for k in range(8)]].mean(0).mean(); fD = fr[ix['dil']].mean(); fR = fr[[ix[f'R{k}'] for k in range(8)]].mean(0).mean()
        out['refit'] = dict(acc=float((rc[0] == y[te]).mean()), flow_T=float(fT), flow_dil=float(fD), flow_R=float(fR),
                            R_fl=sdiv(fT, fD), Q_rad_fl=sdiv(fR, fT))
    return out, reps


def analyze(a):
    from scipy import stats
    from interp.jedi_core import sample
    _, y, _ = sample(per_class=PER, seed=SEED); y = np.asarray(y)
    p = np.random.default_rng(SEED).permutation(len(y)); te = np.sort(p[len(y) // 2:])
    rows = []
    for g in ('jedi', 'flow', 's64', 's32'):
        f = OUT / f'x13_{g}.json'
        if f.exists():
            G = json.loads(f.read_text()); rows += [dict(r, group=g) for r in G['rows']]; log(g, len(G['rows']), 'rows')
        else: log('missing', f)
    I_te = np.random.default_rng(131330).integers(0, len(te), (B, len(te))); I_all = np.random.default_rng(131332).integers(0, len(y), (B, len(y)))
    S = dict(sample=dict(per_class=PER, seed=SEED, N=len(y), N_test=len(te)), B=B, c=C, nets={})
    REP = {}
    for r in rows:
        st, rp = net_stats(r['name'], y, te, I_te, I_all); S['nets'][r['name']] = dict(meta={k: v for k, v in r.items() if k != 'quick'}, **st); REP[r['name']] = rp
    tr = [n for n, v in S['nets'].items() if v['meta']['trained']]
    log('trained', len(tr), 'untrained', len(S['nets']) - len(tr))
    off = S['nets'].get('jedi-3f-n64', {}).get('meta', {}).get('acc'); S['official_n64_acc'] = off; log('official n64 acc on sample', off)
    # per-network table
    for n in S['nets']:
        v = S['nets'][n]; L = v['last']; c = v['calls']
        log(f'{n:30s} acc {v["meta"]["acc"]:.3f} R² {v["meta"]["readout_r2"]["last"]:.2f} | R {L["R"]} {L["ci"]["R"]} Q_rad {L["Q_rad"]} {L["ci"]["Q_rad"]} '
            f'Q_pair {L["Q_pair"]} odd/even {L["odd_over_even"]} rot/dil {L["rot_over_dil"]} | R_fl {c["R_fl"]} Q_rad_fl {c["Q_rad_fl"]} '
            f'flowT {c["flow"]["T"]:+.3f} Q_pair_cc {c["Q_pair_cc"]}' + (f' | refit {v["refit"]}' if 'refit' in v else ''))
    # Holm per-network tests (trained)
    for key, lay in (('p_T_vs_rot', 'last'), ('p_Qrad_eq1', 'last'), ('p_T_ne0', 'last')):
        ps = [S['nets'][n][lay][key] for n in tr]; h = holm([1.0 if q is None else q for q in ps])
        for n, q in zip(tr, h): S['nets'][n][lay][key + '_holm'] = float(q)
    ps = [S['nets'][n]['calls']['p_flowT_vs_rot'] for n in tr]; h = holm(ps)
    for n, q in zip(tr, h): S['nets'][n]['calls']['p_flowT_vs_rot_holm'] = float(q)
    # across-network Spearman (trained), bootstrap over networks and jets
    def gv(n, lay, k): v = S['nets'][n][lay][k] if lay != 'calls' else S['nets'][n]['calls'][k]; return np.nan if v is None else v
    acc = np.array([S['nets'][n]['meta']['acc'] for n in tr]); r2l = np.array([S['nets'][n]['meta']['readout_r2']['last'] for n in tr])
    rng = np.random.default_rng(131340); AC = {}
    pairs = dict(R_vs_Qrad=('last', 'R', 'last', 'Q_rad'), R_vs_Rrad=('last', 'R', 'last', 'R_rad'), R_vs_Qpair=('last', 'R', 'last', 'Q_pair'),
                 Rfl_vs_Qradfl=('calls', 'R_fl', 'calls', 'Q_rad_fl'), Rpool_vs_Qradpool=('pool', 'R', 'pool', 'Q_rad'),
                 R_vs_Rfl=('last', 'R', 'calls', 'R_fl'), R_vs_Rpairabs=('last', 'R', 'last', 'Rpair_abs'),
                 R_vs_Rrotabs=('last', 'R', 'last', 'Rrot_abs'))
    # sensitivity: drop networks whose calls are degenerate (max call share > 0.3, i.e. one class takes > 30 % of calls)
    use = [n for n in tr if S['nets'][n]['meta']['max_call_share'] <= 0.3]; S['excluded_degenerate'] = [n for n in tr if n not in use]
    pairs.update({k + '__nondegenerate': v for k, v in list(pairs.items()) if k in ('R_vs_Qrad', 'R_vs_Rrad', 'R_vs_Rpairabs')})
    for nm, (l1, k1, l2, k2) in pairs.items():
        TR = use if nm.endswith('__nondegenerate') else tr
        acc = np.array([S['nets'][n]['meta']['acc'] for n in TR]); r2l = np.array([S['nets'][n]['meta']['readout_r2']['last'] for n in TR])
        A_ = np.array([gv(n, l1, k1) for n in TR]); B_ = np.array([gv(n, l2, k2) for n in TR]); rho = spearman(A_, B_)
        RA = np.stack([REP[n][l1][k1] for n in TR], 1); RB = np.stack([REP[n][l2][k2] for n in TR], 1); bs = []
        for b in range(B):
            j = rng.integers(0, len(TR), len(TR)); bs.append(spearman(RA[b, j], RB[b, j]))
        ok = np.isfinite(A_) & np.isfinite(B_); pr = stats.spearmanr(A_[ok], B_[ok]).pvalue if ok.sum() > 3 else None
        perm = [spearman(A_[ok], rng.permutation(B_[ok])) for _ in range(10000)]
        pperm = float((1 + np.sum(np.abs(perm) >= abs(rho) - 1e-12)) / 10001)
        part = partial_spearman(A_, B_, np.c_[acc, r2l])
        pb = []
        for b in range(B):
            j = rng.integers(0, len(TR), len(TR)); pb.append(partial_spearman(RA[b, j], RB[b, j], np.c_[acc[j], r2l[j]]))
        AC[nm] = dict(rho=rho, ci=ci(bs), n=int(ok.sum()), p_perm=pperm, p_asym=pr, partial_given_acc_r2=part, partial_ci=ci(pb))
        log(f'Spearman {nm}: {rho:+.3f} {ci(bs)} (n {ok.sum()}, perm p {pperm:.4f}); partial | acc, R² {part:+.3f} {ci(pb)}')
    acc = np.array([S['nets'][n]['meta']['acc'] for n in tr]); r2l = np.array([S['nets'][n]['meta']['readout_r2']['last'] for n in tr])
    Rl = np.array([gv(n, 'last', 'R') for n in tr])
    for nm, v in (('R_vs_acc', acc), ('R_vs_readoutR2', r2l)):
        AC[nm] = dict(rho=spearman(Rl, v), p_perm=float((1 + sum(abs(spearman(Rl, rng.permutation(v))) >= abs(spearman(Rl, v)) for _ in range(10000))) / 10001))
        log(f'Spearman {nm}: {AC[nm]}')
    S['across'] = AC
    # variance decomposition of log R (read-out, last layer) by family; permutation of family labels
    fam = np.array([S['nets'][n]['meta']['family'] for n in tr]); cls = np.array([S['nets'][n]['meta']['cls'] for n in tr])
    def eta2(v, lab):
        g = {l: v[lab == l] for l in np.unique(lab)}; sst = ((v - v.mean()) ** 2).sum()
        ssb = sum(len(x) * (x.mean() - v.mean()) ** 2 for x in g.values()); return float(ssb / sst) if sst > 0 else np.nan
    VD = {}
    for key, lay, k in (('logR_readout', 'last', 'R'), ('logR_flow', 'calls', 'R_fl'), ('logR_pool', 'pool', 'R'), ('R_readout_linear', 'last', 'R')):
        lin = key.endswith('linear'); v = np.array([gv(n, lay, k) for n in tr]); ok = np.isfinite(v) & ((v > 0) | lin)
        tf_ = (lambda q: q) if lin else np.log
        multi = np.array([ok[i] and ((fam == fam[i]) & ok).sum() >= 2 for i in range(len(tr))])
        lv, lf = tf_(v[multi]), fam[multi]; e = eta2(lv, lf); pe = [eta2(lv, rng.permutation(lf)) for _ in range(10000)]
        e_c = eta2(tf_(v[ok]), cls[ok]); pc = [eta2(tf_(v[ok]), rng.permutation(cls[ok])) for _ in range(10000)]
        within_sd = float(np.sqrt(np.mean([lv[lf == f].var(ddof=1) for f in np.unique(lf)]))) if len(lf) else None
        VD[key] = dict(n_nets=int(multi.sum()), n_families=int(len(np.unique(lf))), excluded_nonpositive=[tr[i] for i in np.flatnonzero(~ok)],
                       between_share=e, perm_p=float((1 + np.sum(np.array(pe) >= e - 1e-12)) / 10001), perm_q95=float(np.quantile(pe, .95)),
                       within_family_sd_logR=within_sd, total_sd_logR=float(lv.std(ddof=1)) if len(lv) > 1 else None,
                       class_between_share=e_c, class_perm_p=float((1 + np.sum(np.array(pc) >= e_c - 1e-12)) / 10001),
                       class_perm_q95=float(np.quantile(pc, .95)), n_classes=int(len(np.unique(cls[ok]))))
        log(f'variance decomposition {key}: {VD[key]}')
    S['variance'] = VD
    # paired seed differences (same jets): R(s_a) - R(s_b) per family
    PD = {}
    for f in np.unique(fam):
        ms = sorted([n for n in tr if S['nets'][n]['meta']['family'] == f], key=lambda n: S['nets'][n]['meta']['seed'])
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                a_, b_ = ms[i], ms[j]; ra, rb = gv(a_, 'last', 'R'), gv(b_, 'last', 'R')
                db = REP[a_]['last']['R'] - REP[b_]['last']['R']
                PD[f'{a_} - {b_}'] = dict(family=f, R_a=ra, R_b=rb, diff=ra - rb, ci=ci(db), p=bp(db),
                                          logratio=float(np.log(ra / rb)) if ra > 0 and rb > 0 else None)
    hp = holm([v['p'] if v['p'] is not None else 1.0 for v in PD.values()])
    for (k, v), q in zip(PD.items(), hp): v['p_holm'] = float(q)
    for k, v in PD.items(): log(f'seed pair {k}: ΔR {v["diff"]:+.3f} {v["ci"]} p {v["p"]} Holm {v["p_holm"]:.3g}')
    S['seed_pairs'] = PD
    # untrained references per architecture
    U = {}
    for n, v in S['nets'].items():
        if v['meta']['trained']: continue
        f = v['meta']['family']; U.setdefault(f, []).append(dict(name=n, R=v['last']['R'], Q_rad=v['last']['Q_rad'], r2=v['meta']['readout_r2']['last'],
                                                                 max_call_share=v['meta']['max_call_share'], refit=v.get('refit')))
    S['untrained'] = U
    for f, L in U.items():
        log(f'untrained {f}: ' + ' | '.join(f'R {x["R"]} Q_rad {x["Q_rad"]} R² {x["r2"]:.2f} const {x["max_call_share"]:.2f} refit R_fl '
                                           f'{(x["refit"] or {}).get("R_fl")}' for x in L))
    fo = OUT / 'x13_summary.json'; fo.write_text(json.dumps(clean(S), indent=1)); log('wrote', fo)
    plot(S, tr)


def plot(S, tr):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    col = {'jedi-linear': '#1f77b4', 'jedi-linear-float': '#17becf', 'dsets': '#2ca02c', 'jedis': '#98df8a', 'linf': '#ff7f0e',
           'tf': '#d62728', 'pfn': '#9467bd', 'efn': '#8c564b'}
    lab = {'jedi-linear': 'JEDI-linear (HGQ)', 'jedi-linear-float': 'JEDI-linear float', 'dsets': 'Deep Sets', 'jedis': 'JEDI-style Deep Sets',
           'linf': 'Linformer', 'tf': 'transformer', 'pfn': 'PFN', 'efn': 'EFN'}
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.2))
    seen = set()
    for n in tr:
        v = S['nets'][n]; L = v['last']; c = v['meta']['cls']
        if L['R'] is None or L['Q_rad'] is None: continue
        ax[0].errorbar(L['Q_rad'], L['R'], xerr=np.abs(np.array([L['ci']['Q_rad']]).T - L['Q_rad']) if None not in L['ci']['Q_rad'] else None,
                       yerr=np.abs(np.array([L['ci']['R']]).T - L['R']) if None not in L['ci']['R'] else None, fmt='o', ms=5, color=col.get(c, 'k'),
                       ecolor=col.get(c, 'k'), alpha=.8, elinewidth=.8, label=lab.get(c, c) if c not in seen else None); seen.add(c)
    ax[0].axhline(1, color='gray', lw=.6, ls=':'); ax[0].axvline(1, color='gray', lw=.6, ls=':')
    A = S['across']['R_vs_Qrad']
    ax[0].set_xlabel('Q_rad = read-out response to radial-only edit / to translation')
    ax[0].set_ylabel('R = read-out response to 0.06 rad translation / to matched dilation')
    ax[0].set_title(f'Spearman {A["rho"]:+.2f} [{A["ci"][0]:+.2f}, {A["ci"][1]:+.2f}] over {A["n"]} trained networks', fontsize=10)
    ax[0].set_xscale('symlog', linthresh=0.5); ax[0].set_yscale('symlog', linthresh=0.5); ax[0].legend(fontsize=7, loc='best')
    fams = sorted({S['nets'][n]['meta']['family'] for n in tr}, key=lambda f: np.median([S['nets'][n]['last']['R'] or 0 for n in tr if S['nets'][n]['meta']['family'] == f]))
    for i, f in enumerate(fams):
        ms = [n for n in tr if S['nets'][n]['meta']['family'] == f]
        for n in ms:
            L = S['nets'][n]['last']
            if L['R'] is None: continue
            ax[1].errorbar(i, L['R'], yerr=np.abs(np.array([L['ci']['R']]).T - L['R']) if None not in L['ci']['R'] else None, fmt='o', ms=4,
                           color=col.get(S['nets'][n]['meta']['cls'], 'k'), elinewidth=.8)
        u = [x['R'] for x in S['untrained'].get(f, []) if x['R'] is not None]
        if u: ax[1].plot([i] * len(u), u, 'x', color='gray', ms=4)
    ax[1].set_xticks(range(len(fams))); ax[1].set_xticklabels(fams, rotation=75, fontsize=6.5); ax[1].set_yscale('symlog', linthresh=0.5)
    ax[1].axhline(1, color='gray', lw=.6, ls=':')
    V = S['variance']['logR_readout']
    ax[1].set_ylabel('R (read-out, translation / matched dilation)')
    ax[1].set_title(f'per family (dots: trained seeds, x: untrained); between-family share of var(log R) {V["between_share"]:.2f}, '
                    f'perm. p {V["perm_p"]:.3g}', fontsize=9)
    fig.tight_layout(); fo = OUT / 'x13_axis_size.png'; fig.savefig(fo, dpi=130); log('wrote', fo)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--group', default=''); ap.add_argument('--analyze', action='store_true')
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    analyze(a) if a.analyze else run_group(a)
