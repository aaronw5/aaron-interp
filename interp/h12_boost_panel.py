"""H12 (loop D; claims D1, D8, D4m, new sub-claim D9): do jet taggers read jet mass as a boost-invariant quantity, or as a
combination of jet pT and constituent opening angles fixed by their training pT range?

Edits (on every particle's four-vector; lam = the factor applied to the jet pT; m = invariant jet mass of the particles):
  B  exact Lorentz boost along the jet three-momentum (E-scheme sum of the particles in the table) with rapidity zeta chosen so
     that |P_jet| -> lam |P_jet| (pT -> lam pT, direction unchanged); m and every particle mass unchanged exactly; opening angles
     shrink ~ 1/lam. The stored axis is boosted with the same transformation (as a massless direction) and the inputs are
     measured about it.
  P  pT-only scaling: every four-vector x lam (eta, phi fixed): pT -> lam pT, m -> lam m (the D1 / H3 edit).
  A  angle-only scaling: (deta, dphi) about the stored axis x 1/lam at fixed pT and particle mass: m -> ~ m / lam.
  R  control: rotation of all particles about the stored axis in the (deta, dphi) plane by an angle of random sign whose
     pT-weighted mean displacement equals that of B on the same jet (no change of pT, m, m_A or any IRC-safe shape).
  T  control: translation of all particles by a random-direction (deta, dphi) vector of the same length as B's pT-weighted mean
     displacement (axis kept: changes m_A, not m). For Top / QG the networks recompute the axis from the particles, so T is
     close to a symmetry there.
  To first order B = P(lam) followed by A(lam); so a network that reads a boost-invariant mass gives Delta_B ~ 0 while
  Delta_P ~ -Delta_A; one that reads pT gives Delta_B ~ Delta_P; one that reads angles at fixed pT gives Delta_B ~ Delta_A.
Datasets, samples, models (seed 301 everywhere)
  hls4ml   interp.jedi_core.sample(per_class=4000, seed=301) jets (h1_irc_panel.hls_sample: same indices), 20,000 jets, 64 slots;
           particles massless at (eta, phi) = (deta, dphi) (jet placed at eta = 0 by a longitudinal boost, which leaves pT,
           deta, dphi of massless particles unchanged). --part hls_jedi (jax): JEDI-linear official 3-feature n64, n32, n16;
           jedi_n64_perm = n64 with the entries of every kernel / bias / gamma permuted (seed 7). --part hls_small (conda):
           dsets-d64, linf-2b-d32-k8, tf-2b-d64-e100, jedis-d64 (float JEDI-style) seed 0; EFN (H1, IRC safe, not boost
           invariant); h12 jnorm retrains when present.
  JetClass random rows (seed 301) of each class's first test file, 4,000 per class = 40,000 jets (h1_irc_panel.jc_sample),
           built with eta = jet_eta + sign(jet_eta) deta; ParT_full, ParT_kin, ParticleNet_full, ParticleNet_kin (GPU).
  Top      Top Landscape test split, 20,000 top + 20,000 QCD random rows (rng 301, as data_top.sample); ParT_kin and
           ParticleNet_kin (final H2 weaver checkpoints), LorentzNet (released; rows [::2]), PFN, EFN (H2 energyflow).
  QG       QG test split, 20,000 quark + 20,000 gluon random rows (rng 301); ParT_kin, ParticleNet_kin (final), EFN (H5 torch).
  lam: hls4ml / JetClass {0.8, 0.9, 1.1, 1.25}; Top / QG additionally {0.95, 1.05} (the training pT windows 550-650 /
  500-550 GeV are narrower than the other lam steps).
Metrics (per model x edit x lam): % calls changed (argmax) with a 95 % bootstrap interval over jets; accuracy after; class flow
  matrix; mean Delta log-odds for each binary task with a 95 % bootstrap interval; signal / background efficiency at the
  unedited eps_S = 0.3 / 0.5 thresholds; the same on the jets whose edited jet pT stays inside the training window
  (in_window); exact McNemar of call changes B vs R, P vs R, A vs R, B vs P, B vs A (paired, same jets); Holm across models
  x lam per family. Decomposition: per model, Delta log-softmax of B regressed on those of P and A (no intercept, pooled over
  lam and classes; 95 % bootstrap over jets), and rms ratios. References per jet: invariant m, m_A = sqrt(M0 M2) about the
  stored axis, sum pT, N; mass-window taggers (m and m_A, signal quartiles = eps_S 0.5) before / after; tau21, tau32, D2 on
  1,000 jets (rows [::k]) for B, P, A at lam 0.9 / 1.1 and R 0.9; tau32 cut at eps_S 0.3 / 0.5 for top. C2ST per edit (HGB on
  log sum pT, m, m_A, N, pT-weighted mean dR, unedited vs edited jets on disjoint halves; 2-fold AUC).
Run
  ./job.sh d-h12-jedi 1 2 'python -m interp.h12_boost_panel --part hls_jedi'
  AI_PY=conda ./job.sh d-h12-small 1 2 'python -m interp.h12_boost_panel --part hls_small'
  ./part_job.sh d-h12-jc 'python -m interp.h12_boost_panel --part jetclass'
  ./part_job.sh d-h12-tq 'python -m interp.h12_boost_panel --part top && python -m interp.h12_boost_panel --part qg'
  ./job.sh d-h12-sum 'python -m interp.h12_boost_panel --part summary'
  -> $AI_RESULTS/physics/h12_<part>.json, h12_<part>_perjet.npz, h12_summary.json, h12_boost.png; --smoke N: N jets per class
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results'))
OUT = RES / 'physics'
V = Path('/j-jepa-vol/aaron-interp')
SEED = 301
NBOOT = 1000
T0 = time.time()
LAM4 = (0.8, 0.9, 1.1, 1.25)
LAM6 = (0.8, 0.9, 0.95, 1.05, 1.1, 1.25)


def log(*a):
    print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ================================================================================================================ table + edits
# T = dict(pt, eta, phi, m: (J, L) float64 (pt = 0 padding); ax_eta, ax_phi: (J,) stored axis; ext: {key: (J, L)})
def four(T):
    real = T['pt'] > 0; pt, eta, phi, m = T['pt'], T['eta'], T['phi'], T['m']
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    E = np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2)
    return [np.where(real, a, 0.0) for a in (px, py, pz, E)]


def jetp4(T):
    return np.stack([a.sum(1) for a in four(T)], 1)


def jet_obs(T):
    """invariant m, jet pT (|sum pT vector|), sum pT, N, m_A = sqrt(M0 M2) about the stored axis, pT-weighted mean dR"""
    p = jetp4(T); real = T['pt'] > 0
    m = np.sqrt(np.maximum(p[:, 3] ** 2 - (p[:, :3] ** 2).sum(1), 0)); jpt = np.hypot(p[:, 0], p[:, 1])
    de = np.where(real, T['eta'] - T['ax_eta'][:, None], 0); dp = np.where(real, wrap(T['phi'] - T['ax_phi'][:, None]), 0)
    r2 = de ** 2 + dp ** 2; M0 = T['pt'].sum(1); M2 = (T['pt'] * r2).sum(1)
    return dict(m=m, jpt=jpt, sumpt=M0, N=real.sum(1).astype(float), mA=np.sqrt(M0 * M2),
                mdr=(T['pt'] * np.sqrt(r2)).sum(1) / np.maximum(M0, 1e-12))


def copyT(T):
    U = {k: T[k].copy() for k in ('pt', 'eta', 'phi', 'm', 'ax_eta', 'ax_phi')}; U['ext'] = T['ext']; return U


def _from_p(px, py, pz):
    pt = np.hypot(px, py); eta = np.arcsinh(pz / np.maximum(pt, 1e-300)); phi = np.arctan2(py, px)
    return pt, eta, phi


def boost(T, lam):
    """exact boost along the jet momentum with |P| -> lam |P| (see module docstring); returns (table, info)"""
    real = T['pt'] > 0; px, py, pz, E = four(T)
    P = np.stack([px.sum(1), py.sum(1), pz.sum(1)], 1); EJ = E.sum(1); pa = np.linalg.norm(P, axis=1)
    mJ = np.sqrt(np.maximum(EJ ** 2 - pa ** 2, 0)); mJ = np.maximum(mJ, 1e-6)
    n = P / pa[:, None]; zeta = np.arcsinh(lam * pa / mJ) - np.arcsinh(pa / mJ)
    g, bg = np.cosh(zeta)[:, None], np.sinh(zeta)[:, None]
    ppar = px * n[:, 0:1] + py * n[:, 1:2] + pz * n[:, 2:3]
    dpar = (g - 1) * ppar + bg * E
    qx, qy, qz = px + dpar * n[:, 0:1], py + dpar * n[:, 1:2], pz + dpar * n[:, 2:3]
    U = copyT(T); pt, eta, phi = _from_p(qx, qy, qz)
    U['pt'] = np.where(real, pt, 0); U['eta'] = np.where(real, eta, 0); U['phi'] = np.where(real, phi, 0)
    # stored axis as a massless direction (E = 1)
    a = np.stack([np.cos(T['ax_phi']), np.sin(T['ax_phi']), np.sinh(T['ax_eta'])], 1) / np.cosh(T['ax_eta'])[:, None]
    apar = (a * n).sum(1); da = (g[:, 0] - 1) * apar + bg[:, 0] * 1.0; a2 = a + da[:, None] * n
    _, U['ax_eta'], U['ax_phi'] = _from_p(a2[:, 0], a2[:, 1], a2[:, 2])
    return U, dict(zeta=zeta)


def pscale(T, lam):
    U = copyT(T); U['pt'] = T['pt'] * lam; U['m'] = T['m'] * lam; return U, {}


def rel(T):
    real = T['pt'] > 0
    return np.where(real, T['eta'] - T['ax_eta'][:, None], 0), np.where(real, wrap(T['phi'] - T['ax_phi'][:, None]), 0)


def set_rel(T, de, dp):
    real = T['pt'] > 0
    T['eta'] = np.where(real, T['ax_eta'][:, None] + de, 0); T['phi'] = np.where(real, wrap(T['ax_phi'][:, None] + dp), 0)


def ascale(T, lam):
    U = copyT(T); de, dp = rel(T); set_rel(U, de / lam, dp / lam); return U, {}


def displacement(T, U):
    """pT-weighted (unedited pT) mean distance in the (deta, dphi) plane between each particle's position about its table's axis"""
    d0, p0 = rel(T); d1, p1 = rel(U); w = T['pt']
    return (w * np.hypot(d1 - d0, p1 - p0)).sum(1) / np.maximum(w.sum(1), 1e-12)


def rotate(T, D, rng):
    """rotation about the axis with pT-weighted mean displacement D (J,); random sign"""
    de, dp = rel(T); r = np.hypot(de, dp); w = T['pt']; rbar = (w * r).sum(1) / np.maximum(w.sum(1), 1e-12)
    th = 2 * np.arcsin(np.clip(D / np.maximum(2 * rbar, 1e-12), 0, 1)) * rng.choice([-1.0, 1.0], len(D))
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    U = copyT(T); set_rel(U, c * de - s * dp, s * de + c * dp); return U, dict(theta=th)


def translate(T, D, rng):
    a = rng.uniform(0, 2 * np.pi, len(D)); de, dp = rel(T)
    U = copyT(T); real = T['pt'] > 0
    set_rel(U, np.where(real, de + (D * np.cos(a))[:, None], 0), np.where(real, dp + (D * np.sin(a))[:, None], 0)); return U, {}


def sortT(T):
    o = np.argsort(-T['pt'], 1, kind='stable'); tk = lambda a: np.take_along_axis(a, o, 1)
    U = {k: tk(T[k]) for k in ('pt', 'eta', 'phi', 'm')}; U['ext'] = {k: tk(v) for k, v in T['ext'].items()}
    U['ax_eta'], U['ax_phi'] = T['ax_eta'], T['ax_phi']; return U


def edit_sets(T, lams, seed):
    """yields (name, kind, lam, table) for every edit; the R / T controls are matched to B's displacement on each jet"""
    rng = np.random.default_rng(seed + 1000)
    for lam in lams:
        B, _ = boost(T, lam); D = displacement(T, B)
        yield f'B{lam}', 'B', lam, B, D
        yield f'P{lam}', 'P', lam, pscale(T, lam)[0], None
        yield f'A{lam}', 'A', lam, ascale(T, lam)[0], None
        yield f'R{lam}', 'R', lam, rotate(T, D, rng)[0], None
        yield f'T{lam}', 'T', lam, translate(T, D, rng)[0], None


# ================================================================================================================ statistics
def boot_frac(x, rng, B=NBOOT):
    x = np.asarray(x, bool); n = len(x)
    if n == 0: return [None, None]
    b = rng.binomial(n, x.mean(), B) / n
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def boot_mean(x, rng, B=NBOOT):
    x = np.asarray(x, float); n = len(x)
    if n == 0: return None, [None, None]
    i = rng.integers(0, n, (B, n)); m = x[i].mean(1)
    return float(x.mean()), np.percentile(m, [2.5, 97.5]).tolist()


def binom_p(k, n):
    if n == 0: return 1.0
    k = min(k, n - k)
    lp = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]
    m = max(lp); return float(min(1.0, 2 * math.exp(m) * sum(math.exp(v - m) for v in lp)))


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); out = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * ps[i])); out[i] = run
    return out.tolist()


def logsoft(L):
    L = L - L.max(1, keepdims=True); return L - np.log(np.exp(L).sum(1, keepdims=True))


def logodds(L, sig, bkg):
    ls = logsoft(L); s = ls[:, [sig]]; b = ls[:, bkg]
    return (s[:, 0] - np.log(np.exp(b).sum(1)))


def mcnemar(a, b):
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    return dict(n_first_only=n10, n_second_only=n01, p=binom_p(n10, n10 + n01))


def c2st(F0, F1, seed=0):
    """AUC (2-fold) of a classifier separating unedited (half 1) from edited (half 2) jets"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed); J = len(F0); p = rng.permutation(J); h = J // 2
    X = np.concatenate([F0[p[:h]], F1[p[h:2 * h]]]); y = np.r_[np.zeros(h), np.ones(h)]
    q = rng.permutation(len(y)); X, y = X[q], y[q]; k = len(y) // 2; s = np.zeros(len(y))
    for tr, te in ((slice(0, k), slice(k, None)), (slice(k, None), slice(0, k))):
        c = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(X[tr], y[tr]); s[te] = c.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, s))


def feats_c2st(O):
    return np.stack([np.log(np.maximum(O['sumpt'], 1e-6)), O['m'], O['mA'], O['N'], O['mdr']], 1)


# ================================================================================================================ panel driver
def run_panel(name, T0, y, models, prep, classes, tasks, lams, window, smoke, extra_info=None, obs_k=None, top_task=None):
    """models: {name: (f(prep_out) -> logits (J, C), rows or None)}; tasks: {task: (sig, [bkg])}"""
    rng = np.random.default_rng(SEED); J = len(y); C = len(classes)
    res = dict(part=name, N=J, seed=SEED, classes=classes, labels=np.bincount(y, minlength=C).tolist(), lams=list(lams),
               window=window, tasks={k: [classes[v[0]], [classes[b] for b in v[1]]] for k, v in tasks.items()},
               models={}, edits={}, refs={}, tests={})
    if extra_info: res.update(extra_info)
    O0 = jet_obs(T0); NPZ = dict(y=y.astype(np.int8), m0=O0['m'].astype(np.float32), jpt0=O0['jpt'].astype(np.float32),
                                mA0=O0['mA'].astype(np.float32))
    res['refs']['unedited'] = {k: dict(mean=float(v.mean()), q=np.percentile(v, [0.5, 25, 50, 75, 99.5]).tolist()) for k, v in O0.items()}
    X0 = prep(sortT(T0)); L0, rowsM = {}, {}
    for mn, (f, rows) in models.items():
        t = time.time(); rr = np.arange(J) if rows is None else rows; rowsM[mn] = rr
        L0[mn] = f(X0, rr); c0 = L0[mn].argmax(1); yy = y[rr]
        r = dict(n=int(len(rr)), acc=float((c0 == yy).mean()), acc_ci=boot_frac(c0 == yy, rng),
                 per_class={classes[k]: float((c0[yy == k] == k).mean()) for k in range(C) if (yy == k).any()}, edits={})
        from sklearn.metrics import roc_auc_score
        r['auc'] = {}
        for tn, (s, b) in tasks.items():
            sel = np.isin(yy, [s] + list(b))
            if sel.sum() > 10 and (yy[sel] == s).any(): r['auc'][tn] = float(roc_auc_score(yy[sel] == s, logodds(L0[mn], s, b)[sel]))
        res['models'][mn] = r; NPZ[f'{mn}__base'] = L0[mn].astype(np.float32)
        log(f'{name} {mn}: N {len(rr)} accuracy {r["acc"]:.4f} AUC {r["auc"]} ({time.time() - t:.0f} s)')
    # classical mass-window taggers (eps_S 0.5 = signal quartiles) on unedited jets
    win = {}
    for tn, (s, b) in tasks.items():
        for v in ('m', 'mA'):
            q = np.percentile(O0[v][y == s], [25, 75]); win[f'{tn}|{v}'] = q
    res['refs']['mass_windows'] = {k: v.tolist() for k, v in win.items()}
    thr = {}
    for mn in models:
        rr = rowsM[mn]; yy = y[rr]
        for tn, (s, b) in tasks.items():
            lo = logodds(L0[mn], s, b); thr[(mn, tn)] = {e: float(np.quantile(lo[yy == s], 1 - e)) for e in (0.3, 0.5)}
    CH = {}; DL = {}
    obs_rows = None if obs_k is None else np.arange(J)[::obs_k]
    if obs_rows is not None:
        try:
            from interp.h1_irc_panel import observables as h1obs
            def shapes(T):
                de, dp = rel(T); S = sortT(dict(pt=T['pt'][obs_rows], eta=de[obs_rows], phi=dp[obs_rows], m=T['m'][obs_rows], ext={},
                                                ax_eta=T['ax_eta'][obs_rows], ax_phi=T['ax_phi'][obs_rows]))
                o = h1obs(dict(pt=S['pt'], eta=S['eta'], phi=S['phi'], m=S['m'], ext={}))
                return {k: o[k] for k in ('tau21', 'tau32', 'D2', 'mSD')}
            t = time.time(); SH0 = shapes(T0); log(f'shapes on {len(obs_rows)} unedited jets ({time.time() - t:.0f} s)')
        except Exception as e:
            import traceback; traceback.print_exc(); obs_rows = None; res['shapes_error'] = repr(e)
    for en, kind, lam, U, D in edit_sets(T0, lams, SEED):
        t = time.time(); O1 = jet_obs(U)
        inw = (O1['jpt'] >= window[0]) & (O1['jpt'] <= window[1])
        e = dict(kind=kind, lam=lam, in_window_frac=float(inw.mean()),
                 rel_change={k: dict(mean=float(np.mean(O1[k] / np.maximum(O0[k], 1e-12) - 1)),
                                     absmax=float(np.max(np.abs(O1[k] / np.maximum(O0[k], 1e-12) - 1))))
                             for k in ('m', 'jpt', 'sumpt', 'mA', 'mdr')})
        if D is not None: e['displacement'] = dict(mean=float(D.mean()), median=float(np.median(D)))
        e['classical'] = {}
        for tn, (s, b) in tasks.items():
            for v in ('m', 'mA'):
                q = win[f'{tn}|{v}']; p0 = (O0[v] > q[0]) & (O0[v] < q[1]); p1 = (O1[v] > q[0]) & (O1[v] < q[1]); bk = np.isin(y, b)
                e['classical'][f'{tn}|{v}_window'] = dict(sig_eff=[float(p0[y == s].mean()), float(p1[y == s].mean())],
                                                         bkg_eff=[float(p0[bk].mean()), float(p1[bk].mean())],
                                                         calls_changed=float((p0 != p1).mean()))
        try:
            cr = np.random.default_rng(SEED + 7).permutation(J)[:20000]
            e['c2st_auc'] = c2st(feats_c2st(O0)[cr], feats_c2st(O1)[cr], seed=SEED)
        except Exception as ex:
            e['c2st_auc'] = None; e['c2st_error'] = repr(ex)
        if obs_rows is not None and en in ('B0.9', 'B1.1', 'P0.9', 'A0.9', 'R0.9', 'T0.9'):
            SH1 = shapes(U); e['shapes'] = {}
            for k in SH0:
                d = SH1[k] - SH0[k]; mu, ci = boot_mean(d, rng)
                e['shapes'][k] = dict(mean_delta=mu, ci=ci, median_abs_rel=float(np.median(np.abs(d) / np.maximum(np.abs(SH0[k]), 1e-12))))
            if top_task is not None:
                s = tasks[top_task][0]; ys = y[obs_rows]
                for eps in (0.3, 0.5):
                    c = np.quantile(SH0['tau32'][ys == s], eps); p0 = SH0['tau32'] < c; p1 = SH1['tau32'] < c; bk = np.isin(ys, tasks[top_task][1])
                    e['shapes'][f'tau32_cut_eps{eps}'] = dict(sig_eff=[float(p0[ys == s].mean()), float(p1[ys == s].mean())],
                                                             bkg_eff=[float(p0[bk].mean()), float(p1[bk].mean())])
        X1 = prep(sortT(U)); e['models'] = {}
        for mn, (f, rows) in models.items():
            rr = rowsM[mn]; L1 = f(X1, rr); c0, c1 = L0[mn].argmax(1), L1.argmax(1); yy = y[rr]; ch = c0 != c1; iw = inw[rr]
            st = dict(changed=float(ch.mean()), ci=boot_frac(ch, rng), acc_after=float((c1 == yy).mean()),
                      in_window=dict(n=int(iw.sum()), changed=float(ch[iw].mean()) if iw.any() else None, ci=boot_frac(ch[iw], rng)))
            F = np.zeros((C, C), int); np.add.at(F, (c0, c1), 1); st['flow'] = F.tolist()
            st['own_class_call'] = {classes[k]: [float((c0[yy == k] == k).mean()), float((c1[yy == k] == k).mean())]
                                    for k in range(C) if (yy == k).any()}
            st['tasks'] = {}
            for tn, (s, b) in tasks.items():
                d = logodds(L1, s, b) - logodds(L0[mn], s, b); sel = np.isin(yy, [s] + list(b)); mu, ci = boot_mean(d[sel], rng)
                lo1 = logodds(L1, s, b); w = {}
                for eps, th in thr[(mn, tn)].items():
                    w[f'eps{eps}'] = dict(sig_eff=float((lo1[yy == s] > th).mean()), bkg_eff_before=float((logodds(L0[mn], s, b)[np.isin(yy, b)] > th).mean()),
                                          bkg_eff_after=float((lo1[np.isin(yy, b)] > th).mean()))
                ms, cis = boot_mean(d[yy == s], rng); mb, cib = boot_mean(d[np.isin(yy, b)], rng)
                st['tasks'][tn] = dict(dlogodds=mu, ci=ci, dlogodds_sig=ms, ci_sig=cis, dlogodds_bkg=mb, ci_bkg=cib, wp=w)
            e['models'][mn] = st; CH[(mn, en)] = ch
            DL[(mn, en)] = (logsoft(L1) - logsoft(L0[mn])).astype(np.float32)
            NPZ[f'{mn}__{en}'] = c1.astype(np.int8)
        NPZ[f'jpt__{en}'] = O1['jpt'].astype(np.float32); NPZ[f'm__{en}'] = O1['m'].astype(np.float32); NPZ[f'mA__{en}'] = O1['mA'].astype(np.float32)
        res['edits'][en] = e
        log(f'{name} {en}: dm {e["rel_change"]["m"]["mean"]:+.2e} (max {e["rel_change"]["m"]["absmax"]:.1e}) djpt {e["rel_change"]["jpt"]["mean"]:+.3f} '
            f'dmA {e["rel_change"]["mA"]["mean"]:+.4f} in-window {e["in_window_frac"]:.3f} c2st {e["c2st_auc"]} | '
            + ' '.join(f'{mn} {e["models"][mn]["changed"]:.4f}' for mn in models) + f' ({time.time() - t:.0f} s)')
    # paired tests and decomposition
    pairs = [('B', 'R'), ('P', 'R'), ('A', 'R'), ('B', 'T'), ('B', 'P'), ('B', 'A')]
    tests = []
    for mn in models:
        for lam in lams:
            for a, b in pairs:
                k = f'{mn}|{a}{lam}_vs_{b}{lam}'; t_ = mcnemar(CH[(mn, f'{a}{lam}')], CH[(mn, f'{b}{lam}')])
                t_['diff'] = float(CH[(mn, f'{a}{lam}')].mean() - CH[(mn, f'{b}{lam}')].mean())
                n = len(CH[(mn, f'{a}{lam}')]); i = rng.integers(0, n, (NBOOT, n))
                t_['diff_ci'] = np.percentile(CH[(mn, f'{a}{lam}')][i].mean(1) - CH[(mn, f'{b}{lam}')][i].mean(1), [2.5, 97.5]).tolist()
                res['tests'][k] = t_; tests.append((k, f'{a}_vs_{b}', t_['p']))
    for fam in {t[1] for t in tests}:
        sel = [t for t in tests if t[1] == fam]; hp = holm([t[2] for t in sel])
        for (k, _, _), h in zip(sel, hp): res['tests'][k]['holm_part'] = h
    res['decomposition'] = {}
    for mn in models:
        db = np.concatenate([DL[(mn, f'B{l}')] for l in lams]); dp_ = np.concatenate([DL[(mn, f'P{l}')] for l in lams])
        da = np.concatenate([DL[(mn, f'A{l}')] for l in lams]); dr = np.concatenate([DL[(mn, f'R{l}')] for l in lams])
        nj = len(rowsM[mn])
        use = [np.sqrt((dp_ ** 2).mean()) > 1e-6, np.sqrt((da ** 2).mean()) > 1e-6]   # a regressor with no response is left out (coef 0)

        def fit(idx):
            Y = db[idx].ravel(); X = np.stack([dp_[idx].ravel(), da[idx].ravel()], 1); coef = np.zeros(2)
            if any(use):
                c, *_ = np.linalg.lstsq(X[:, use], Y, rcond=None); coef[np.array(use)] = c
            r2 = 1 - ((Y - X @ coef) ** 2).sum() / max((Y ** 2).sum(), 1e-30)
            return coef, r2
        coef, r2 = fit(np.arange(len(db)))
        bs = []
        for _ in range(200):
            j = rng.integers(0, nj, nj); idx = (j[None, :] + nj * np.arange(len(lams))[:, None]).ravel(); bs.append(fit(idx)[0])
        bs = np.array(bs); rms = lambda a: float(np.sqrt((a ** 2).mean()))
        res['decomposition'][mn] = dict(a_P=float(coef[0]), a_P_ci=np.percentile(bs[:, 0], [2.5, 97.5]).tolist(), b_A=float(coef[1]),
                                        b_A_ci=np.percentile(bs[:, 1], [2.5, 97.5]).tolist(), r2=float(r2),
                                        regressors_used=dict(P=bool(use[0]), A=bool(use[1])),
                                        corr_P_A=float(np.corrcoef(dp_.ravel(), da.ravel())[0, 1]) if all(use) else None,
                                        rms=dict(B=rms(db), P=rms(dp_), A=rms(da), R=rms(dr), PplusA=rms(dp_ + da), B_minus_P=rms(db - dp_),
                                                 B_minus_A=rms(db - da), B_minus_PplusA=rms(db - dp_ - da)),
                                        per_lam={str(l): dict(B=rms(DL[(mn, f'B{l}')]), P=rms(DL[(mn, f'P{l}')]), A=rms(DL[(mn, f'A{l}')]),
                                                              R=rms(DL[(mn, f'R{l}')]), T=rms(DL[(mn, f'T{l}')])) for l in lams})
        d = res['decomposition'][mn]
        log(f'{name} {mn} decomposition: B = {d["a_P"]:.3f} P + {d["b_A"]:.3f} A (R2 {d["r2"]:.3f}); rms B {d["rms"]["B"]:.3f} P {d["rms"]["P"]:.3f} '
            f'A {d["rms"]["A"]:.3f} R {d["rms"]["R"]:.3f}')
    return res, NPZ


def save(res, NPZ, part, smoke):
    OUT.mkdir(parents=True, exist_ok=True); sfx = '_smoke' if smoke else ''
    p = OUT / f'h12_{part}{sfx}.json'; p.write_text(json.dumps(res, indent=1, default=float))
    np.savez_compressed(OUT / f'h12_{part}{sfx}_perjet.npz', **NPZ); log(f'wrote {p}')


# ================================================================================================================ hls4ml
HLS_CL = ['g', 'q', 'W', 'Z', 't']
HLS_TASKS = {'W_vs_QCD': (2, [0, 1]), 'Z_vs_QCD': (3, [0, 1]), 't_vs_QCD': (4, [0, 1]), 'Z_vs_W': (3, [2])}


def hls_load(per):
    from interp.h1_irc_panel import hls_sample
    T, y = hls_sample(per, SEED)
    T = dict(pt=T['pt'], eta=T['eta'], phi=T['phi'], m=T['m'], ax_eta=T['ax_eta'], ax_phi=T['ax_phi'], ext={})
    return T, y


def hls_prep(T):
    de, dp = rel(T); real = T['pt'] > 0
    return np.stack([np.where(real, T['pt'], 0), de, dp], -1)[:, :64].astype(np.float32)


def hls_window(T):
    jpt = jet_obs(T)['jpt']; return [float(np.percentile(jpt, 0.5)), float(np.percentile(jpt, 99.5))]


def permute_weights(model, seed=7):
    """null: the entries of every kernel / bias / BN gamma permuted within the tensor (one rng, model.weights order)"""
    rng = np.random.default_rng(seed)
    for w in model.weights:
        if getattr(w, 'path', w.name).endswith(('/kernel', '/bias', '/gamma')):
            v = np.asarray(w.numpy()); w.assign(rng.permutation(v.ravel()).reshape(v.shape).astype(v.dtype))


def part_hls_jedi(smoke):
    from interp.jedi_core import Jedi
    from jetdistill import network as N
    T0, y = hls_load(smoke or 4000)
    nets = {f'jedi_n{n}': Jedi(n, '3-feature') for n in (64, 32, 16)}
    orig = N.redraw_untrained; N.redraw_untrained = permute_weights
    try: nets['jedi_n64_perm'] = Jedi(64, '3-feature', untrained=True, seed=7)
    finally: N.redraw_untrained = orig
    models = {k: ((lambda x, rr, n=n: n.run(x[rr])['logits']), None) for k, n in nets.items()}
    known = json.loads((V / 'data' / 'official_acc.json').read_text())['3-feature'] if (V / 'data' / 'official_acc.json').exists() else {}
    res, NPZ = run_panel('hls_jedi', T0, y, models, hls_prep, HLS_CL, HLS_TASKS, LAM4, hls_window(T0), smoke,
                         dict(sample=f'interp.jedi_core.sample(per_class={smoke or 4000}, seed={SEED})',
                              known_acc={f'jedi_n{n}': known.get(str(n)) for n in (64, 32, 16)}), obs_k=20, top_task='t_vs_QCD')
    save(res, NPZ, 'hls_jedi', smoke)


def part_hls_small(smoke):
    import torch
    from interp.h1_irc_panel import load_small, torch_logits, efn_class
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
    SMALL = V / 'models' / 'small'
    T0, y = hls_load(smoke or 4000); models = {}; known = {}
    runs = [('dsets', 'dsets-d64-n64'), ('linf', 'linf-2b-d32-k8-n64'), ('tf_d64', 'tf-2b-d64-n64-e100'), ('jedis', 'jedis-d64-n64')]
    for tag, run in runs:
        net, sh, sc = load_small(SMALL / run / 'seed0'); models[tag] = ((lambda x, rr, net=net, sh=sh, sc=sc: torch_logits(net, x[rr], sh, sc)), None)
        known[tag] = json.loads((SMALL / run / 'seed0' / 'test.json').read_text())['accuracy']
    # pT-normalised / pT-flattened retrains (interp.h12_jnorm_train), when finished
    from interp import h12_jnorm_train as JN
    for d in sorted((V / 'models' / 'small').glob('jedis-d64-n64-h12*/seed*')):
        if (d / 'test.json').exists():
            net, prep_fn = JN.load(d); tag = d.parent.name.replace('jedis-d64-n64-', '') + '_' + d.name
            models[tag] = ((lambda x, rr, net=net, prep_fn=prep_fn: JN.logits(net, prep_fn, x[rr])), None)
            known[tag] = json.loads((d / 'test.json').read_text())['accuracy']
    efn = efn_class()(); efn.load_state_dict(torch.load(OUT / 'h1_efn' / 'best.pt', map_location='cpu')); efn.eval()
    models['efn'] = ((lambda x, rr: torch_logits(efn, x[rr], np.zeros(3, np.float32), np.ones(3, np.float32))), None)
    known['efn'] = json.loads((OUT / 'h1_efn' / 'info.json').read_text()).get('test_acc')
    res, NPZ = run_panel('hls_small', T0, y, models, hls_prep, HLS_CL, HLS_TASKS, LAM4, hls_window(T0), smoke,
                         dict(sample=f'interp.jedi_core.sample(per_class={smoke or 4000}, seed={SEED})', known_acc=known))
    save(res, NPZ, 'hls_small', smoke)


# ================================================================================================================ JetClass
def part_jetclass(smoke):
    from interp.h1_irc_panel import jc_sample, jc_table, pn_logits
    from interp.part_core import ParTNetwork, CL, inputs, build
    from interp.p1v_verify_part import fwd
    from interp.pn_net import load_particlenet
    from jetdistill.config import MODELS
    dev = os.environ.get('JETDISTILL_DEVICE', 'cpu')
    J0 = jc_sample(smoke or 4000, SEED); y = J0['label'].astype(int); Tj = jc_table(J0)
    T0 = dict(pt=Tj['pt'], eta=Tj['eta'], phi=Tj['phi'], m=Tj['m'], ax_eta=Tj['ax_eta'], ax_phi=Tj['ax_phi'], ext=Tj['ext'])

    def prep(T):
        Jx = dict(J0); Jx['jet_eta'] = T['ax_eta'].astype(np.float32); Jx['jet_phi'] = T['ax_phi'].astype(np.float32)
        sg = np.where(Jx['jet_eta'] < 0, -1.0, 1.0)[:, None]; de, dp = rel(T)
        return build(Jx, T['pt'], sg * de, dp, T['m'], extra=T['ext'])
    from interp.part_core import kin
    Jb = prep(sortT(T0)); Jref = build(J0, *kin(J0))
    rt = max(float(np.abs(Jb[k] - Jref[k]).max()) for k in ('px', 'py', 'pz', 'energy', 'deta', 'dphi'))
    log(f'round trip table -> build vs build(kin(J0)): max abs diff {rt:.2e}')
    sub = lambda J, rr: {k: (v[rr] if np.ndim(v) else v) for k, v in J.items()}
    models = {}
    for n in ('full', 'kin'):
        net = ParTNetwork(n); models[f'ParT_{n}'] = ((lambda J, rr, net=net, n=n: fwd(net, *inputs(sub(J, rr), n))[1].astype(np.float64)), None)
    for n, dims in (('full', 17), ('kin', 7)):
        pn = load_particlenet(Path(MODELS).parent / 'particlenet' / f'ParticleNet_{n}.pt', dims, dev)
        models[f'PN_{n}'] = ((lambda J, rr, pn=pn, n=n: pn_logits(pn, sub(J, rr), n)), None)
    i = CL.index
    tasks = {'W_vs_QCD': (i('Wqq'), [i('QCD')]), 'Z_vs_QCD': (i('Zqq'), [i('QCD')]), 'Top_vs_QCD': (i('Tbqq'), [i('QCD')]),
             'Hbb_vs_QCD': (i('Hbb'), [i('QCD')]), 'Z_vs_W': (i('Zqq'), [i('Wqq')])}
    res, NPZ = run_panel('jetclass', T0, y, models, prep, list(CL), tasks, LAM4, [500.0, 1000.0], smoke,
                         dict(sample=f'random rows (seed {SEED}) of each class first test file, {smoke or 4000} per class', roundtrip_maxdiff=rt,
                              known_acc=dict(ParT_full='0.859 (D4) / 0.8605 (H8)', ParT_kin='0.745 (D4) / 0.737 (H8)', PN_full='0.844 published')),
                         obs_k=40, top_task='Top_vs_QCD')
    save(res, NPZ, 'jetclass', smoke)


# ================================================================================================================ Top Landscape
def _rows(y, per, seed, classes=(0, 1)):
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per, replace=False) for c in classes]))


def table_from_full(F, nn):
    """(J, L, >=4) px, py, pz, E (pT-sorted, zero padded) -> table with the E-scheme axis of the particles"""
    P = F[..., :4].astype(np.float64); L = P.shape[1]; real = (np.arange(L)[None, :] < nn[:, None]) & (np.hypot(P[..., 0], P[..., 1]) > 0)
    pt, eta, phi = _from_p(P[..., 0], P[..., 1], P[..., 2])
    m = np.sqrt(np.maximum(P[..., 3] ** 2 - (P[..., :3] ** 2).sum(-1), 0))
    S = (P * real[..., None]).sum(1); _, ae, ap = _from_p(S[:, 0], S[:, 1], S[:, 2])
    return dict(pt=np.where(real, pt, 0), eta=np.where(real, eta, 0), phi=np.where(real, phi, 0), m=np.where(real, m, 0),
                ax_eta=ae, ax_phi=ap, ext={})


def part_top(smoke):
    import torch
    from interp import data_top as dt
    from interp.h4_top_wmass import WeaverNet, LNet, Flow, kin3_of
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    d = dt.DST / 'test'; ylab = np.load(d / 'label.npy'); idx = _rows(ylab, smoke or 20000, SEED)
    t = time.time(); F = np.load(d / 'full.npy')[idx]; jet = np.load(d / 'jet.npy')[idx]; log(f'top test rows {len(idx)} read ({time.time() - t:.0f} s)')
    nn = np.minimum(jet[:, 5].astype(np.int64), F.shape[1]); y = ylab[idx].astype(int); T0 = table_from_full(F, nn); del F

    def prep(T):
        pt, eta, phi, m = T['pt'], T['eta'], T['phi'], T['m']; real = pt > 0
        P4 = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2)], -1) * real[..., None]
        return dict(P4=P4.astype(np.float32), nn=real.sum(1))
    two = lambda s: np.stack([np.zeros_like(s), s], 1)
    models = {}
    for kind, dname in (('ParT', 'part_kin'), ('PN', 'pnet_kin')):
        net = WeaverNet(kind, V / 'models/top' / dname / 'net_best_epoch_state.pt', dev)
        models[f'{kind}_kin'] = ((lambda X, rr, net=net: two(net.score(X['P4'][rr], X['nn'][rr]))), None)
    ln = LNet(dev, mass64=True); rl = np.arange(len(y))[::2]; order = np.random.default_rng(SEED + 5).permutation(len(rl))
    models['LorentzNet'] = ((lambda X, rr: two(ln.score(X['P4'][rr], X['nn'][rr], order))), rl)
    for a in ('pfn', 'efn'):
        fl = Flow(a); models[a.upper()] = ((lambda X, rr, fl=fl: two(fl.score(kin3_of(X['P4'][rr], X['nn'][rr])))), None)
    known = dict(ParT_kin='0.940 (weaver test, H2)', PN_kin='0.940 (weaver test, H2)', PFN=0.9283, EFN=0.9218)
    try: known['LorentzNet'] = json.loads((V / 'models/top/lorentznet/test-result.json').read_text()).get('test_acc')
    except Exception: pass
    res, NPZ = run_panel('top', T0, y, models, prep, ['QCD', 'top'], {'top_vs_QCD': (1, [0])}, LAM6, [550.0, 650.0], smoke,
                         dict(sample=f'Top Landscape test, {smoke or 20000} top + {smoke or 20000} QCD random rows (rng {SEED}); LorentzNet rows [::2]',
                              known_acc=known), obs_k=40, top_task='top_vs_QCD')
    save(res, NPZ, 'top', smoke)


# ================================================================================================================ QG
def part_qg(smoke):
    import torch
    from interp import data_qg as dq
    from interp.h5_qg_counting import Models, make_ef
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    d = dq.DST / 'test'; ylab = np.load(d / 'label.npy'); idx = _rows(ylab, smoke or 20000, SEED)
    t = time.time(); F = np.load(d / 'full.npy')[idx]; pid = np.load(d / 'pid.npy')[idx].astype(np.int64); jet = np.load(d / 'jet.npy')[idx]
    log(f'qg test rows {len(idx)} read ({time.time() - t:.0f} s)')
    nn = np.minimum(jet[:, 5].astype(np.int64), F.shape[1]); y = ylab[idx].astype(int); T0 = table_from_full(F, nn); del F
    T0['m'] = np.zeros_like(T0['m'])           # massless, as the conversion (E = |p|)
    T0['ext'] = dict(pid=pid * (T0['pt'] > 0))

    def prep(T):
        real = T['pt'] > 0
        return dict(pt=T['pt'], eta=T['eta'], phi=T['phi'], pid=T['ext']['pid'].astype(np.int64) * real, n=real.sum(1))
    M = Models(dev)
    for name, arch in (('part_kin', 'part'), ('pnet_kin', 'pn')): M.add_weaver(name, arch, False)
    net = make_ef('efn'); net.load_state_dict(torch.load(V / 'models' / 'qg' / 'h5_efn_torch.pt', map_location='cpu'))
    M.add_torch('efn_t', net, json.loads((V / 'models' / 'qg' / 'h5_efn_torch.json').read_text()))
    sub = lambda X, rr: {k: v[rr] for k, v in X.items()}
    two = lambda s: np.stack([np.zeros_like(s), s], 1)          # class 1 = quark (label 1); Models.logit = log p(g) - log p(q)
    models = {k: ((lambda X, rr, k=k: two(-M.logit(k, sub(X, rr)))), None) for k in ('part_kin', 'pnet_kin', 'efn_t')}
    res, NPZ = run_panel('qg', T0, y, models, prep, ['gluon', 'quark'], {'q_vs_g': (1, [0])}, LAM6, [500.0, 550.0], smoke,
                         dict(sample=f'QG test, {smoke or 20000} quark + {smoke or 20000} gluon random rows (rng {SEED})',
                              known_acc=dict(part_kin='0.827 (weaver test, H2)', pnet_kin='0.827 (weaver test, H2)'), ckpt=M.info), obs_k=40)
    save(res, NPZ, 'qg', smoke)


# ================================================================================================================ summary
def part_summary():
    R = {p: json.loads((OUT / f'h12_{p}.json').read_text()) for p in ('hls_jedi', 'hls_small', 'jetclass', 'top', 'qg') if (OUT / f'h12_{p}.json').exists()}
    S = dict(parts=list(R), accuracy={}, holm_global={}, table={})
    fam_of = lambda k: k.split('|', 1)[1][0] + '_vs_' + k.split('_vs_')[1][0]
    for fam in ('B_vs_R', 'P_vs_R', 'A_vs_R', 'B_vs_P', 'B_vs_A', 'B_vs_T'):
        keys = [(p, k, t['p']) for p, r in R.items() for k, t in r['tests'].items() if fam_of(k) == fam and 'perm' not in k]
        hp = holm([x[2] for x in keys])
        for (p, k, _), h in zip(keys, hp): S['holm_global'][f'{p}|{k}'] = h
    for p, r in R.items():
        for mn, m in r['models'].items():
            S['accuracy'][f'{p}|{mn}'] = dict(acc=m['acc'], auc=m['auc'], n=m['n'])
            row = {}
            for en, e in r['edits'].items():
                row[en] = dict(changed=e['models'][mn]['changed'], ci=e['models'][mn]['ci'], in_window=e['models'][mn]['in_window']['changed'])
            S['table'][f'{p}|{mn}'] = dict(changed=row, decomposition=r['decomposition'][mn])
    # post-hoc: % calls changed on jets whose edited jet pT stays in the central 95 % of the unedited jet-pT distribution
    rng = np.random.default_rng(SEED); S['central95'] = {}
    for p, r in R.items():
        Z = np.load(OUT / f'h12_{p}_perjet.npz'); j0 = Z['jpt0']; lo, hi = np.percentile(j0, [2.5, 97.5]); J = len(j0)
        S['central95'][p] = dict(window=[float(lo), float(hi)], models={})
        for mn in r['models']:
            c0 = Z[f'{mn}__base'].argmax(1); rows = np.arange(J) if len(c0) == J else np.arange(J)[::2]
            out = {}
            for en in r['edits']:
                j1 = Z[f'jpt__{en}'][rows]; inw = (j1 >= lo) & (j1 <= hi); ch = (Z[f'{mn}__{en}'] != c0)
                out[en] = dict(n=int(inw.sum()), changed=float(ch[inw].mean()) if inw.any() else None, ci=boot_frac(ch[inw], rng))
            S['central95'][p]['models'][mn] = out
        log(f'central95 {p}: window {lo:.0f}-{hi:.0f} GeV; ' + ' '.join(f'{mn} B0.9 {S["central95"][p]["models"][mn]["B0.9"]["changed"]}' for mn in r['models']))
    (OUT / 'h12_summary.json').write_text(json.dumps(S, indent=1, default=float)); log('wrote h12_summary.json')
    fig(R)


def fig(R):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    order = [('hls_jedi', 'jedi_n64', 'JEDI-linear n64'), ('hls_jedi', 'jedi_n32', 'JEDI-linear n32'), ('hls_small', 'dsets', 'Deep Sets'),
             ('hls_small', 'tf_d64', 'Transformer'), ('hls_small', 'jedis', 'JEDI-style float'), ('hls_small', 'h12ptnorm_seed0', 'JEDI-style, pT/ΣpT input'),
             ('hls_small', 'h12ptflat_seed0', 'JEDI-style, flat-pT weights'), ('hls_small', 'h12raw_seed0', 'JEDI-style retrain (control)'),
             ('hls_small', 'efn', 'EFN (hls4ml)'),
             ('jetclass', 'ParT_full', 'ParT full'), ('jetclass', 'ParT_kin', 'ParT kin'), ('jetclass', 'PN_full', 'ParticleNet full'),
             ('jetclass', 'PN_kin', 'ParticleNet kin'), ('top', 'ParT_kin', 'Top: ParT'), ('top', 'PN_kin', 'Top: ParticleNet'),
             ('top', 'LorentzNet', 'Top: LorentzNet'), ('top', 'PFN', 'Top: PFN'), ('top', 'EFN', 'Top: EFN'),
             ('qg', 'part_kin', 'QG: ParT'), ('qg', 'pnet_kin', 'QG: ParticleNet'), ('qg', 'efn_t', 'QG: EFN')]
    order = [o for o in order if o[0] in R and o[1] in R[o[0]]['models']]
    lam = 0.9
    kinds = [('P', 'pT x 0.9 (mass x 0.9)', '#d1495b'), ('B', 'boost: pT x 0.9, mass fixed', '#2e5eaa'),
             ('A', 'angles / 0.9 (mass / 0.9)', '#edae49'), ('R', 'control: rotation', '#aaaaaa')]
    fig_, ax = plt.subplots(figsize=(13, 5.2)); w = 0.2
    for b, (k, lab, col) in enumerate(kinds):
        v, lo, hi = [], [], []
        for p, mn, _ in order:
            st = R[p]['edits'][f'{k}{lam}']['models'][mn]; v.append(100 * st['changed']); lo.append(100 * st['ci'][0]); hi.append(100 * st['ci'][1])
        xs = np.arange(len(order)) + (b - 1.5) * w
        ax.bar(xs, v, w, color=col, label=lab); ax.errorbar(xs, v, [np.subtract(v, lo), np.subtract(hi, v)], fmt='none', ecolor='k', lw=0.6)
    ax.set_xticks(np.arange(len(order))); ax.set_xticklabels([o[2] for o in order], rotation=35, ha='right', fontsize=8.5)
    ax.set_ylabel('calls changed (% of sample jets)')
    grp = [o[0] for o in order]; edges = [i for i in range(1, len(grp)) if grp[i] != grp[i - 1] and not (grp[i].startswith('hls') and grp[i - 1].startswith('hls'))]
    for e_ in edges: ax.axvline(e_ - 0.5, color='k', lw=0.6, ls=':')
    ax.set_title('Jet pT x 0.9 by a boost at fixed jet mass changes hls4ml-trained calls as much as pT-only or angle-only scalings; '
                 'JetClass/Top/QG models change less\n(error bars: 95 % bootstrap over jets; grey = rotation control)', fontsize=9.5)
    ax.legend(fontsize=8.5, frameon=False, loc='upper right'); fig_.tight_layout(); fig_.savefig(OUT / 'h12_boost.png', dpi=130)
    log(f'wrote {OUT / "h12_boost.png"}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--part', required=True, choices=['hls_jedi', 'hls_small', 'jetclass', 'top', 'qg', 'summary'])
    ap.add_argument('--smoke', type=int, default=0); a = ap.parse_args()
    log(f'H12 part {a.part} smoke {a.smoke}')
    if a.part == 'summary': part_summary()
    else: dict(hls_jedi=part_hls_jedi, hls_small=part_hls_small, jetclass=part_jetclass, top=part_top, qg=part_qg)[a.part](a.smoke)
    log('done')
