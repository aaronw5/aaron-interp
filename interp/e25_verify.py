"""E25 verification (independent re-implementation of the key edits; new controls).   python -m interp.e25_verify
-> $AI_RESULTS/jedi/e25_verify.json

Sample: rows of interp.jedi_core.sample(1500, 2507) minus the E25 main rows (sample(3000, 2501)) and minus the E25 fresh rows
(sample(1500, 2502)).  All 16 columns from val_c16_f16.npy at those rows (abort unless pt/etarel/phirel equal sample() x).
Edits are written here from scratch (not imported from e25_jedi16), on real slots (pt > 0) only:
  T        rigid +d in eta about the stored axis: etarel += d, eta += d, deltaR = hypot(etarel', phirel), pz = pT sinh(eta'),
           e = sqrt(m^2 + (pT cosh eta')^2) with the particle's own m^2, erel = e / E_J (E_J = median e/erel), costheta = tanh(eta').
  T_dRonly only deltaR set to hypot(etarel + d, phirel).
  T_dR     T with deltaR held.
  R        (etarel, phirel) rotated about the axis by the angle that moves a particle at the pT-weighted mean r by d (random sign);
           eta, phi, px, py, pz, e, erel, costheta recomputed; deltaR unchanged.
  dil_dR   only deltaR multiplied per jet by s_T = sqrt(M2(T)/M2), M2 = sum pT r^2 (r from etarel, phirel).
  jit_dR   (new control) only deltaR changed: r_i -> c r_i exp(sigma eps_i), eps_i = +-1 random, c per jet such that sum pT r'^2 = M2
           (M2 held); sigma set so that the pT-weighted mean |r' - r| over all W jets equals that of T.
  con_dR   (new control) only deltaR divided by s_T (M2 lowered by the same log amount T raises it).
  Dose: d = 0.03 and 0.09 for T_dRonly and the matched dil_dR.
Metric: W->Z = fraction of true W jets (y == 2) called Z (3); flow = W->Z(edit) - W->Z(unedited); 95 % bootstrap over jets (2,000);
McNemar (Z indicator) vs R; ratio flow(T_dRonly)/flow(dil_dR) with a paired bootstrap; Holm over the McNemar tests.
Also: l1 purity null restricted to the units that have any nonzero effective l1 weight (each feature's column permuted among those
units, 20,000 draws; two-sided p), for 16-feature and 16-feature-perminv; untrained 16-feature redraws seeds 3-8 (T_dRonly vs R).
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'; OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)
FEAT16 = ['px', 'py', 'pz', 'e', 'erel', 'pt', 'ptrel', 'eta', 'etarel', 'etarot', 'phi', 'phirel', 'phirot', 'deltaR', 'costheta',
          'costhetarel']
C = {k: i for i, k in enumerate(FEAT16)}
GROUPS = dict(pT_like=['pt', 'ptrel', 'e', 'erel', 'pz'], momentum_xy=['px', 'py'], angle_rel=['etarel', 'phirel'],
              rot_frame=['etarot', 'phirot', 'costhetarel'], radial=['deltaR'], absolute=['eta', 'phi', 'costheta'])
RNG = np.random.default_rng(2508); NB = 2000; B = 2500
from interp.e19_common import mcnemar, holm  # noqa: E402


def rows(y, per, seed):
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per, replace=False) for c in range(5)]))


def boot(x):
    x = np.asarray(x, np.float64); i = RNG.integers(0, len(x), (NB, len(x))); m = x[i].mean(1)
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def boot_ratio(a, b):
    i = RNG.integers(0, len(a), (NB, len(a))); r = a[i].mean(1) / b[i].mean(1)
    return [float(np.quantile(r, .025)), float(np.quantile(r, .975))]


def load():
    from jetdistill.config import RESULTS, DATA
    from interp.jedi_core import sample
    y = np.asarray(np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')['y'])
    idx = rows(y, 1500, 2507); x3, ys, _ = sample(1500, 2507)
    assert np.array_equal(ys, y[idx])
    X = np.asarray(np.load(Path(DATA) / 'val_c16_f16.npy', mmap_mode='r')[idx, :64]).astype(np.float32)
    d = float(np.abs(X[..., [5, 8, 11]] - x3).max()); log('max |16f cols - sample x|', d)
    if d > 1e-3: raise SystemExit('ABORT: columns differ')
    keep = ~np.isin(idx, rows(y, 3000, 2501)) & ~np.isin(idx, rows(y, 1500, 2502))
    return X[keep], ys[keep], d, int(keep.sum())


def base(X):
    Xd = X.astype(np.float64); rl = Xd[..., C['pt']] > 0; pt = Xd[..., C['pt']]
    e, eta = Xd[..., C['e']], Xd[..., C['eta']]
    m2 = np.clip(e ** 2 - (pt * np.cosh(eta)) ** 2, 0, None)
    EJ = np.nanmedian(np.where(rl, e / np.where(rl, Xd[..., C['erel']], 1), np.nan), 1)
    er, pr = Xd[..., C['etarel']], Xd[..., C['phirel']]; r = np.hypot(er, pr)
    M2 = (pt * r ** 2).sum(1)
    return dict(Xd=Xd, rl=rl, pt=pt, m2=m2, EJ=EJ, er=er, pr=pr, r=r, M2=M2)


def put(b, newcols):
    E = b['Xd'].copy()
    for k, v in newcols.items(): E[..., C[k]] = np.where(b['rl'], v, E[..., C[k]])
    return E.astype(np.float32)


def absolute(b, eta2, phi2):
    pt = b['pt']; e = np.sqrt(b['m2'] + (pt * np.cosh(eta2)) ** 2)
    return dict(eta=eta2, phi=phi2, px=pt * np.cos(phi2), py=pt * np.sin(phi2), pz=pt * np.sinh(eta2), e=e, erel=e / b['EJ'][:, None],
                costheta=np.tanh(eta2))


def make_edits(b, W):
    Xd = b['Xd']; pt, er, pr, r = b['pt'], b['er'], b['pr'], b['r']; E = {}; meta = {}
    def T_cols(d):  # noqa: E306
        return dict(etarel=er + d, deltaR=np.hypot(er + d, pr), **absolute(b, Xd[..., C['eta']] + d, Xd[..., C['phi']]))
    def sT(d):  # noqa: E306
        return np.sqrt((pt * ((er + d) ** 2 + pr ** 2)).sum(1) / np.maximum(b['M2'], 1e-30))
    tc = T_cols(0.06)
    E['T'] = put(b, tc)
    E['T_dRonly'] = put(b, dict(deltaR=tc['deltaR']))
    E['T_dR'] = put(b, {k: v for k, v in tc.items() if k != 'deltaR'})
    rbar = (pt * r).sum(1) / np.maximum(pt.sum(1), 1e-12)
    ang = 2 * np.arcsin(np.minimum(1.0, 0.06 / np.maximum(2 * rbar, 1e-12))) * RNG.choice([-1, 1], len(Xd))
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]; er2, pr2 = c * er - s * pr, s * er + c * pr
    phi2 = (Xd[..., C['phi']] + (pr2 - pr) + np.pi) % (2 * np.pi) - np.pi
    E['R'] = put(b, dict(etarel=er2, phirel=pr2, **absolute(b, Xd[..., C['eta']] + (er2 - er), phi2)))
    s6 = sT(0.06); meta['sT_median'] = float(np.median(s6))
    E['dil_dR'] = put(b, dict(deltaR=Xd[..., C['deltaR']] * s6[:, None]))
    E['con_dR'] = put(b, dict(deltaR=Xd[..., C['deltaR']] / s6[:, None]))
    # M2-preserving jitter matched to T's pT-weighted mean |dr| on W jets
    target = (pt * np.abs(tc['deltaR'] - Xd[..., C['deltaR']]) * b['rl'])[W].sum() / pt[W].sum()
    eps = RNG.choice([-1.0, 1.0], r.shape)
    def jit(sig):  # noqa: E306
        rj = r * np.exp(sig * eps); cj = np.sqrt(b['M2'] / np.maximum((pt * rj ** 2).sum(1), 1e-30)); return rj * cj[:, None]
    lo, hi = 0.0, 3.0
    for _ in range(50):
        mid = (lo + hi) / 2; dv = (pt * np.abs(jit(mid) - r) * b['rl'])[W].sum() / pt[W].sum()
        lo, hi = (mid, hi) if dv < target else (lo, mid)
    rj = jit(lo); E['jit_dR'] = put(b, dict(deltaR=rj))
    meta.update(T_mean_abs_dr_W=float(target), jit_sigma=float(lo),
                jit_mean_abs_dr_W=float((pt * np.abs(rj - r) * b['rl'])[W].sum() / pt[W].sum()),
                jit_M2_rel_change_max=float(np.nanmax(np.abs((pt * rj ** 2).sum(1) / np.maximum(b['M2'], 1e-30) - 1)[b['M2'] > 0])),
                T_dlogM2_mean=float(np.log(s6[b['M2'] > 0] ** 2).mean()))
    for d in (0.03, 0.09):
        t = T_cols(d); E[f'T_dRonly_{d}'] = put(b, dict(deltaR=t['deltaR']))
        E[f'dil_dR_{d}'] = put(b, dict(deltaR=Xd[..., C['deltaR']] * sT(d)[:, None]))
    for k, v in E.items():   # columns actually changed
        dd = np.abs(v.astype(np.float64) - Xd).max((0, 1)); meta.setdefault('cols_changed', {})[k] = [f for f in FEAT16 if dd[C[f]] > 1e-6]
        meta.setdefault('pad_changed', {})[k] = bool((v[~b['rl']] != Xd[~b['rl']].astype(np.float32)).any())
    return E, meta


def calls(net, x): return net.run(x, batch=B)['logits'].argmax(1)


def purity_null(net, live_note):
    F = 16; K, _ = net.KB['l1']; K = K.reshape(F, 64)
    # effective weights as in E25: features with any kept slot
    q = (list(net.L['l1'].iq.quantizers) if hasattr(net.L['l1'].iq, 'quantizers') else [net.L['l1'].iq])[0].quantizer
    def slots(a):  # noqa: E306
        a = np.asarray(a, np.float64); a2 = a.reshape(a.shape[-2:]) if a.ndim >= 2 else a.reshape(1, -1); return np.broadcast_to(a2, (64, F))
    bits = np.maximum(slots(q.k) + slots(q.i) + slots(q.f), 0); kept = (bits > 0).any(0)
    nz = (K * kept[:, None]) != 0; act = np.flatnonzero(nz.any(0)); N = nz[:, act]
    gi = [[C[f] for f in fs] for fs in GROUPS.values()]
    def pur(M): return float(np.mean([sum(M[g, j].any() for g in gi) == 1 for j in range(M.shape[1])]))  # noqa: E306
    obs = pur(N); null = []
    for _ in range(20000):
        P = np.array([RNG.permutation(N.shape[1]) for _ in range(F)]); Mp = np.take_along_axis(N, P, 1)
        Mp = Mp[:, Mp.any(0)]; null.append(pur(Mp) if Mp.shape[1] else np.nan)
    null = np.array(null); lo = float((1 + (null <= obs + 1e-12).sum()) / (1 + len(null))); hi = float((1 + (null >= obs - 1e-12).sum()) / (1 + len(null)))
    return dict(n_active=int(len(act)), n_nonzero=int(N.sum()), obs_exact=obs, null_q=[float(np.nanquantile(null, p)) for p in (.025, .5, .975)],
                p_lower=lo, p_upper=hi, p_two_sided=min(1.0, 2 * min(lo, hi)), note=live_note)


def main():
    from interp.jedi_core import Jedi
    X, y, d, N = load(); W = y == 2; R = dict(N=N, N_W=int(W.sum()), class_counts=np.bincount(y, minlength=5).tolist(), max_col_diff=d,
                                            seed='sample(1500, 2507) minus E25 main and fresh rows', procedure=__doc__)
    log(f'N {N}, N_W {int(W.sum())}')
    b = base(X); E, meta = make_edits(b, W); R['edit_meta'] = meta; log('meta', meta)
    del b
    out = {}; tests = []
    for name, fam in (('16f', '16-feature'), ('16f_perminv', '16-feature-perminv'), ('3f', '3-feature')):
        net = Jedi(64, fam); sel = (lambda x: x) if name.startswith('16') else (lambda x: x[..., [5, 8, 11]])
        c0 = calls(net, sel(X)); z0 = (c0[W] == 3).astype(float); d_ = dict(acc=float((c0 == y).mean()), W_to_Z=float(z0.mean()), arms={})
        Z = {}
        for arm, xe in E.items():
            if not name.startswith('16') and arm not in ('T', 'T_dR', 'R'): continue
            ce = calls(net, sel(xe)); Z[arm] = (ce[W] == 3).astype(float)
            d_['arms'][arm] = dict(W_to_Z=float(Z[arm].mean()), W_to_Z_ci=boot(Z[arm]), flow=float((Z[arm] - z0).mean()), flow_ci=boot(Z[arm] - z0),
                                   calls_changed=float((ce != c0).mean()), W_to_q_or_g=float(np.isin(ce[W], [0, 1]).mean()))
        for arm in Z:
            if arm != 'R':
                m = mcnemar(Z['R'].astype(bool), Z[arm].astype(bool)); d_['arms'][arm]['mcnemar_vs_R'] = m; tests.append((name, arm, m['p']))
        if name.startswith('16'):
            for a, bb in (('T_dRonly', 'dil_dR'), ('T', 'dil_dR'), ('T_dRonly_0.03', 'dil_dR_0.03'), ('T_dRonly_0.09', 'dil_dR_0.09')):
                fa, fb = Z[a] - z0, Z[bb] - z0
                d_[f'ratio_{a}_over_{bb}'] = dict(r=float(fa.mean() / fb.mean()) if fb.mean() > 0 else None, ci=boot_ratio(fa, fb) if fb.mean() > 0 else None)
            d_['mcnemar_jit_vs_T_dRonly'] = mcnemar(Z['jit_dR'].astype(bool), Z['T_dRonly'].astype(bool))
            d_['mcnemar_T_vs_T_dRonly'] = mcnemar(Z['T'].astype(bool), Z['T_dRonly'].astype(bool))
            d_['purity_active_null'] = purity_null(net, 'null permutes each feature column among units with any nonzero effective l1 weight')
        out[name] = d_
        log(name, json.dumps({k: (v['W_to_Z'], v['flow_ci']) for k, v in d_['arms'].items()}))
        log(name, {k: v for k, v in d_.items() if k.startswith('ratio') or k.startswith('purity')})
        del net
    ph = holm([t[2] for t in tests])
    for (nm, arm, _), h in zip(tests, ph): out[nm]['arms'][arm]['holm_p_vs_R'] = h
    R['models'] = out
    unt = {}
    for s in range(3, 9):
        net = Jedi(64, '16-feature', untrained=True, seed=s); c0 = calls(net, X)
        cT = calls(net, E['T_dRonly']); cR = calls(net, E['R'])
        unt[s] = dict(n_distinct_calls=int(len(np.unique(c0))), acc=float((c0 == y).mean()), W_to_Z=float((c0[W] == 3).mean()),
                      W_to_Z_T_dRonly=float((cT[W] == 3).mean()), W_to_Z_R=float((cR[W] == 3).mean()),
                      calls_changed_T_dRonly=float((cT != c0).mean()), calls_changed_R=float((cR != c0).mean()))
        log('untrained', s, unt[s]); del net
    R['untrained'] = unt
    import resource; R['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024; R['wall_s'] = time.time() - T0

    def clean(o):
        if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [clean(v) for v in o]
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.bool_): return bool(o)
        return o
    (OUT / 'e25_verify.json').write_text(json.dumps(clean(R), indent=1)); log('done')


main()
