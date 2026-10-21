"""E33x (claims A1x, A10, A3; round 16): do the pT-weighted squared-radius unit and the soft-count unit recur across the
retrained JEDI-linear networks more often than random rank-window hinge formulas that fit the official network equally well,
and more often than in untrained networks?  No training.

  python -m interp.e33x_recurrence_null --part trained [--models m1,m2 --label a]  -> $AI_RESULTS/jedi/e33x_trained[_a].json
  python -m interp.e33x_recurrence_null --part untrained   -> $AI_RESULTS/jedi/e33x_untrained.json
  python -m interp.e33x_recurrence_null --part combine     -> $AI_RESULTS/jedi/e33x.json (merges every e33x_trained*/untrained* file)
  (--smoke: 200 jets / class, 20 decoys, 2 + 2 networks, 50 bootstrap draws)

PLAN (fixed before the first run)
Networks: the 10 trained E33 networks (official 3-feature n64 = reference, n32, n128, perm-inv n64, 16-feature n64; X12
  retrained HGQ-r150 s0-s2, float s0-s1) and untrained redraws of each (jetdistill.network.redraw_untrained: every kernel, bias
  and batch-norm scale redrawn from a Gaussian with that weight's trained mean and s.d.; number formats kept), seeds 0, 1, 2
  (30 untrained networks).  Jets: fit = jedi_core.sample(2000, 4401), test = sample(2000, 4402) (10,000 each; E33v samples),
  loaded with E33's loader (n32 = first 32 slots, n128 = 128-slot copy of the same rows, 16-feature = 16-column copy).
Formulas (each fitted to every live S unit, live = s.d. > 1e-9 on the fit jets; S = second-round pooled objects):
  named roles (E33): hard = sum pT of the k hardest (k fitted); r2 = a + b sum pT r^2 + c sum r^2 + d N(pT < 2, r > 0.1)
    (the pT-weighted squared-radius unit); soft = sum over slots of rank >= k of (t - pT)+ (soft / empty-slot count; t, k
    fitted); core = sum (t - r)+.  Per unit the parameter with the best fit R^2; the 4 roles assigned to distinct units by
    Hungarian matching on fit R^2 (one unit serves one formula); test R^2 of the assigned unit.
  competitors (E33v): total pT, sum r, multiplicity N, sum pT r (single features, all slots).
  decoys: 600 random hinge families f = sum_{rank < k} ReLU(cos th z_pT + sin th z_r + b) (k = 1..n fitted, th ~ U(0, 2 pi),
    b ~ N(0, 1), rng 4433; z_pT, z_r standardised over the real particles of the 64-slot fit jets, the same constants for every
    network) plus E33v's fixed decoys sum_{rank<k} r, sum_{rank<k} pT^2, sum_{rank>=k} (r - t)+.
  Competitors and decoys are assigned as a 5th row in the same Hungarian matching with the 4 roles (they cannot take a unit
  that a role holds unless the summed fit R^2 is higher).  The unassigned best unit is reported too.
Recurrence: a formula qualifies if its assigned unit in the reference reaches test R^2 >= 0.9.  Recurrence count = number of
  the other 9 trained networks in which its assigned unit reaches test R^2 >= 0.9.  H0: a named formula recurs no more than
  qualifying decoys.  p = (1 + #{qualifying decoys with count >= named count}) / (1 + #qualifying); share with Clopper-Pearson
  95 % interval.  Architectural base rate: the same count over the 30 untrained networks; trained (k / 9) vs untrained
  (u / 30) one-sided Fisher exact test.  Per network: p_fm = (1 + #{qualifying decoys with test R^2 in m >= formula's}) /
  (1 + #qualifying); Holm over 8 formulas x 9 networks.  Also recurrence among decoys split by |corr| with the reference r2
  unit's formula value (>= 0.9 'r2-like' vs < 0.9).
Radial weight w(r) on each trained network's r2-assigned unit: per jet one real particle with r in bin [0.015 or lo, hi),
  bins 0-0.05, 0.05-0.1, 0.1-0.15, 0.15-0.2, 0.2-0.3, 0.3-0.4; edits: radial move r -> r +- 0.01 (deta, dphi scaled; pT, rank
  and azimuth kept) and the control: azimuthal move of the same arc length (0.01) at fixed r and pT (a radial formula predicts
  0).  D = (S(r + 0.01) - S(r - 0.01)) / 0.02 per particle; in each bin OLS D = alpha_b + gamma_b pT; for w(r) = r^beta per
  GeV (unit ~ sum pT r^beta), gamma_b = C beta r_b^(beta - 1): beta from a weighted grid fit (beta in [-1, 4], step 0.01, C
  closed form, weights 1 / se_b^2) on the bin means r_b; 1,000-draw particle bootstrap within bins for the CI; H0 beta = 2
  (two-sided bootstrap p).  Control ratio = mean |azimuthal response| / mean |radial response| (per 0.01 move), bootstrap CI.
Matched hard / soft pT-add control on each trained network's hard-assigned unit (pT-sum unit): per jet the soft particle =
  the real particle of rank >= 14 with the largest gap to the next-harder one; delta = min(1 GeV, 0.9 gap) (>= 0.05 GeV, rank
  kept) added to it; the same delta added to the hardest particle.  Response per GeV beta = sum dS delta / sum delta^2;
  ratio soft / hard (total-pT unit: 1; hard-window unit with k <= 14: 0), 1,000-draw jet bootstrap, H0 ratio = 1.
z windows: E33 step 5 on the test jets (all particles' (deta, dphi) x 1.05 / 0.95, R = dz / d ln m, window centre = mass where
  the 10 GeV bin mean of R crosses from > 0 to <= 0 with |mean / se| >= 3 on both sides within 30 GeV).  Statistic T = number
  of (reference window, other trained network) pairs with a window centre within +-10 GeV.  Positional null: every other
  network's centres shifted by its own offset ~ U(-50, 50) GeV, 1,000 draws; p = (1 + #{T_null >= T}) / 1001.  Same T for
  the untrained networks (per network, scaled to 9 networks).
"""
import argparse, contextlib, gc, json, os, sys, time
from pathlib import Path
import numpy as np

os.environ.setdefault('KERAS_BACKEND', 'jax')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import e33_program_recurrence_jedi as E  # noqa: E402

OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
T0 = time.time()
ROLES = E.ROLES
COMPS = ['total_pT', 'sum_r', 'mult_N', 'sum_pT_r']
FORMULAS = ROLES + COMPS
BINS = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.15), (0.15, 0.2), (0.2, 0.3), (0.3, 0.4)]
DR = 0.01
SEEDS_U = (0, 1, 2)


def log(*a): print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


@contextlib.contextmanager
def redrawn(seed):
    """keras.models.load_model returns a model with weights redrawn (untrained control) while active"""
    import keras
    from jetdistill.network import redraw_untrained
    orig = keras.models.load_model

    def f(*a, **k):
        m = orig(*a, **k); redraw_untrained(m, seed); return m
    keras.models.load_model = f
    try: yield
    finally: keras.models.load_model = orig


# ------------------------------------------------------------------------------------------------ data cache
_DATA = {}


def data(per, n, f16):
    key = (n, f16)
    if key not in _DATA:
        xf, xkf, x3f, yf, _ = E.load(per, 4401, n, f16); xt, xkt, x3t, yt, _ = E.load(per, 4402, n, f16)
        _DATA[key] = dict(xf=xf, xkf=xkf, yf=yf, xt=xt, xkt=xkt, yt=yt, x3f=x3f, x3t=x3t, pf=E.particles(xkf), pt=E.particles(xkt))
    return _DATA[key]


def zstats(per):
    d = data(per, 64, False); p, r, re_, _ = d['pf']
    return (float(p[re_].mean()), float(p[re_].std()), float(r[re_].mean()), float(r[re_].std()))


def hinge(P, fam, zs):
    pt, r, real, _ = P; th, b = fam
    zp = (pt - zs[0]) / zs[1]; zr = (r - zs[2]) / zs[3]
    return np.cumsum(real * np.maximum(np.cos(th) * zp + np.sin(th) * zr + b, 0), 1)


def fixed_decoys(P, n):
    p_, r_, re_, _ = P; rank = np.arange(r_.shape[1])[None]
    outer = np.concatenate([(np.maximum(r_ - t, 0) * re_ * (rank >= k)).sum(1)[:, None]
                            for t in (0.05, 0.1, 0.2, 0.3, 0.4) for k in E.K_SOFT if k < n], 1)
    return dict(sum_r_prefix=np.cumsum(r_, 1), sum_pT2_prefix=np.cumsum(p_ ** 2, 1), outer_ramp=outer)


def comp_feats(P):
    p_, r_, re_, _ = P; rf = re_.astype(float)
    return dict(total_pT=p_.sum(1)[:, None], sum_r=r_.sum(1)[:, None], mult_N=rf.sum(1)[:, None], sum_pT_r=(p_ * r_).sum(1)[:, None])


def fam_unit(Zf, Zt, Ff, Ft):
    """per unit: best fit R^2 over the family's columns, test R^2 at that column, column index"""
    r2f, r2t, _, _ = E.single_fit(Zf, Zt, Ff, Ft); ar = np.arange(Zf.shape[1]); kb = np.argmax(r2f, 0)
    return r2f[kb, ar], r2t[kb, ar], kb


def hung(rows_fit, rows_test, extra=None):
    """Hungarian assignment of formula rows to distinct units on fit R^2 -> list of (unit or None, test R^2 or None) per row"""
    from scipy.optimize import linear_sum_assignment
    F = np.array(rows_fit, float); T = np.array(rows_test, float)
    if F.shape[1] == 0: return [(None, None)] * len(F)
    Fm = np.where(np.isfinite(F), F, -1.0); ri, ci = linear_sum_assignment(-Fm); a = dict(zip(ri, ci))
    return [(int(a[q]), float(T[q, a[q]]) if np.isfinite(T[q, a[q]]) else None) if q in a else (None, None) for q in range(len(F))]


def unassigned(f, t):
    if len(f) == 0: return None
    i = int(np.argmax(np.where(np.isfinite(f), f, -9))); return float(t[i]) if np.isfinite(t[i]) else None


# ------------------------------------------------------------------------------------------------ one network
def fits(model, d, n, fams, zs):
    St = model.run(d['xt'], keep=('S', 'logits')); Sf = model.run(d['xf'], keep=('S',))['S']
    acc = float((St['logits'].argmax(1) == d['yt']).mean())
    live = [c for c in range(64) if Sf[:, c].std() > 1e-9]; Zf, Zt = Sf[:, live], St['S'][:, live]
    R = dict(acc=acc, acc_ci=E.wilson(int(round(acc * len(d['yt']))), len(d['yt'])), n_live_S=len(live), live_S=live)
    RF, RT = E.role_feats(d['xkf']), E.role_feats(d['xkt'])
    rows_f, rows_t, rows_k = [], [], []
    for role in ROLES:
        if role == 'r2':
            if len(live):
                f, t, _ = E.fit_set(Zf, Zt, RF['r2'][0][1], RT['r2'][0][1]); k = np.zeros(len(live), int)
            else: f = t = np.zeros(0); k = np.zeros(0, int)
        else:
            Ff = np.concatenate([g for _, g in RF[role]], 1); Ft = np.concatenate([g for _, g in RT[role]], 1)
            f, t, k = fam_unit(Zf, Zt, Ff, Ft) if len(live) else (np.zeros(0), np.zeros(0), np.zeros(0, int))
        rows_f.append(f); rows_t.append(t); rows_k.append(k)
    asg = hung(rows_f, rows_t); R['roles'] = {}
    for q, role in enumerate(ROLES):
        u, t = asg[q]
        R['roles'][role] = dict(unit=None if u is None else f'S{live[u]}', unit_idx=u, r2_test=t,
                                param=None if u is None else RF[role][int(rows_k[q][u])][0],
                                best_unassigned=unassigned(rows_f[q], rows_t[q]))
    def as_row(Ff, Ft):
        if not len(live): return None, None
        f, t, _ = fam_unit(Zf, Zt, Ff, Ft); a = hung(rows_f + [f], rows_t + [t])[-1]
        return a[1], unassigned(f, t)
    cf, ct = comp_feats(d['pf']), comp_feats(d['pt']); R['competitors'] = {}
    for c in COMPS:
        h, b = as_row(cf[c], ct[c]); R['competitors'][c] = dict(r2_test=h, best_unassigned=b)
    dec_h, dec_b = [], []
    for fam in fams:
        h, b = as_row(hinge(d['pf'], fam, zs), hinge(d['pt'], fam, zs)); dec_h.append(h); dec_b.append(b)
    fdf, fdt = fixed_decoys(d['pf'], n), fixed_decoys(d['pt'], n)
    for nm in fdf:
        h, b = as_row(fdf[nm], fdt[nm]); dec_h.append(h); dec_b.append(b)
    R['decoys_hung'] = dec_h; R['decoys_best'] = dec_b
    return R, Sf, St['S'], live


def radial_weight(model, d, c, f16, rng, nboot):
    """w(r) on S unit c: radial +-0.01 and azimuthal arc 0.01 one-particle moves per r bin"""
    xk = d['xkt']; pt, r, real, _ = d['pt']; out = dict(bins=[])
    S0 = model.run(d['xt'], keep=('S',))['S'][:, c]
    for lo, hi in BINS:
        j = E.pick(real & (r >= max(lo, 0.015)) & (r < hi), rng); I = np.flatnonzero(j >= 0); jj = j[I]; a = np.arange(len(I))
        if len(I) < 30: out['bins'].append(dict(lo=lo, hi=hi, n=len(I))); continue
        x0 = xk[I].astype(np.float64); r0 = r[I, jj]; p0 = pt[I, jj]
        res = {}
        for nm, s in (('plus', DR), ('minus', -DR)):
            e = x0.copy(); e[a, jj, 1:3] *= ((r0 + s) / r0)[:, None]; e = e.astype(np.float32)
            xin = E.to16(d['xt'][I], xk[I], e) if f16 else e
            res[nm] = model.run(xin, keep=('S',))['S'][:, c]
        e = x0.copy(); th = DR / r0; de, dp = e[a, jj, 1].copy(), e[a, jj, 2].copy()
        e[a, jj, 1] = np.cos(th) * de - np.sin(th) * dp; e[a, jj, 2] = np.sin(th) * de + np.cos(th) * dp; e = e.astype(np.float32)
        xin = E.to16(d['xt'][I], xk[I], e) if f16 else e
        az = (model.run(xin, keep=('S',))['S'][:, c] - S0[I]) / DR
        D = (res['plus'] - res['minus']) / (2 * DR); radp = (res['plus'] - S0[I]) / DR
        out['bins'].append(dict(lo=lo, hi=hi, n=len(I), r_mean=float(r0.mean()), pT_mean=float(p0.mean()), _D=D, _pT=p0, _az=az, _radp=radp))
    def ols(D, p):
        A = np.column_stack([np.ones(len(D)), p]); w, *_ = np.linalg.lstsq(A, D, rcond=None)
        res = D - A @ w; s2 = (res ** 2).sum() / max(len(D) - 2, 1); cov = s2 * np.linalg.pinv(A.T @ A)
        return w, np.sqrt(np.diag(cov))
    ok = [b for b in out['bins'] if '_D' in b]
    if len(ok) < 3: out['beta'] = None; return out
    rb = np.array([b['r_mean'] for b in ok]); G = []; SE = []
    for b in ok:
        w, se = ols(b['_D'], b['_pT']); b['alpha'], b['gamma'], b['gamma_se'] = float(w[0]), float(w[1]), float(se[1])
        b['mean_D'] = float(b['_D'].mean()); b['mean_abs_radial'] = float(np.abs(b['_radp']).mean()); b['mean_abs_azimuthal'] = float(np.abs(b['_az']).mean())
        G.append(w[1]); SE.append(se[1])
    G, SE = np.array(G), np.maximum(np.array(SE), 1e-12); W = 1 / SE ** 2; grid = np.arange(-1, 4.0001, 0.01)

    def fitbeta(g):
        best = (np.inf, None, None)
        for bt in grid:
            f = bt * rb ** (bt - 1); den = (W * f * f).sum()
            if den <= 0: continue
            C = (W * g * f).sum() / den; sse = (W * (g - C * f) ** 2).sum()
            if sse < best[0]: best = (sse, bt, C)
        return best
    sse, bhat, C = fitbeta(G); bs = []; rat = []
    for _ in range(nboot):
        g = []; num = 0.0; den = 0.0
        for b in ok:
            q = rng.integers(0, b['n'], b['n']); w, _ = ols(b['_D'][q], b['_pT'][q]); g.append(w[1])
            num += np.abs(b['_az'][q]).sum(); den += np.abs(b['_radp'][q]).sum()
        bs.append(fitbeta(np.array(g))[1]); rat.append(num / den if den > 0 else np.nan)
    bs = np.array(bs, float); rat = np.array(rat)
    num = sum(np.abs(b['_az']).sum() for b in ok); den = sum(np.abs(b['_radp']).sum() for b in ok)
    out.update(beta=float(bhat), beta_ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], C=float(C), chi2=float(sse), n_bins=len(ok),
               p_beta_eq_2=float(min(1.0, 2 * min((1 + (bs <= 2).sum()) / (1 + nboot), (1 + (bs >= 2).sum()) / (1 + nboot)))),
               az_over_radial=float(num / den) if den > 0 else None,
               az_over_radial_ci=[float(np.nanquantile(rat, .025)), float(np.nanquantile(rat, .975))],
               gamma_sign=[int(np.sign(g)) for g in G])
    for b in ok:
        for k in ('_D', '_pT', '_az', '_radp'): b.pop(k)
    return out


def hard_soft(model, d, c, f16, rng, nboot):
    xk = d['xkt']; pt, r, real, _ = d['pt']; J, n = pt.shape; rank = np.arange(n)[None]
    S0 = model.run(d['xt'], keep=('S',))['S'][:, c]
    prev = np.concatenate([np.full((J, 1), np.inf), pt[:, :-1]], 1); gap = np.where(real & (rank >= 14), prev - pt, -1.0)
    j = np.argmax(gap, 1); g = gap[np.arange(J), j]; dl = np.minimum(1.0, 0.9 * g); I = np.flatnonzero(dl >= 0.05)
    if len(I) < 30: return dict(n=len(I))
    a = np.arange(len(I)); x0 = xk[I].astype(np.float64); dlt = dl[I]; resp = {}
    for nm, col in (('soft', j[I]), ('hard', np.zeros(len(I), int))):
        e = x0.copy(); e[a, col, 0] += dlt; e = e.astype(np.float32)
        xin = E.to16(d['xt'][I], xk[I], e) if f16 else e
        resp[nm] = model.run(xin, keep=('S',))['S'][:, c] - S0[I]
    b = lambda o, q: float((o[q] * dlt[q]).sum() / (dlt[q] ** 2).sum())  # noqa: E731
    al = np.arange(len(I)); bh, bsft = b(resp['hard'], al), b(resp['soft'], al); bs = []
    for _ in range(nboot):
        q = rng.integers(0, len(I), len(I)); hh = b(resp['hard'], q); bs.append(b(resp['soft'], q) / hh if hh != 0 else np.nan)
    bs = np.array(bs)
    return dict(n=len(I), mean_delta=float(dlt.mean()), mean_soft_rank=float(j[I].mean()), beta_hard_per_GeV=bh, beta_soft_per_GeV=bsft,
                ratio=bsft / bh if bh != 0 else None, ratio_ci=[float(np.nanquantile(bs, .025)), float(np.nanquantile(bs, .975))],
                p_ratio_eq_1=float(min(1.0, 2 * min((1 + np.sum(bs <= 1)) / (1 + nboot), (1 + np.sum(bs >= 1)) / (1 + nboot)))))


def run_net(name, seed, per, fams, zs, nboot, edits):
    kind, fam, n = E.MODELS[name]
    ctx = redrawn(seed) if seed is not None else contextlib.nullcontext()
    with ctx:
        model = E.Model(name)
    d = data(per, n, model.f16)
    R, Sf, Stt, live = fits(model, d, n, fams, zs); R.update(model=name, untrained_seed=seed)
    log(f'{name} seed {seed}: acc {R["acc"]:.4f} {R["acc_ci"]}; live S {R["n_live_S"]}; roles '
        + ', '.join(f'{k} {v["unit"]} {v["param"]} R2 {v["r2_test"]}' for k, v in R['roles'].items())
        + '; comps ' + ', '.join(f'{k} {v["r2_test"]}' for k, v in R['competitors'].items())
        + f'; decoys (assigned) R2 >= 0.9: {sum((v or -1) >= 0.9 for v in R["decoys_hung"])}/{len(R["decoys_hung"])}')
    if name == 'official_n64' and seed is None and R['roles']['r2']['unit_idx'] is not None:
        # |corr| between each decoy's best-k column and the r2 formula fitted to the reference r2 unit (test jets)
        u = live[R['roles']['r2']['unit_idx']]; RF, RT = E.role_feats(d['xkf']), E.role_feats(d['xkt'])
        A = np.column_stack([np.ones(len(d['yf'])), RF['r2'][0][1]]); w, *_ = np.linalg.lstsq(A, Sf[:, u], rcond=None)
        fr2 = np.column_stack([np.ones(len(d['yt'])), RT['r2'][0][1]]) @ w; cs = []
        def cmax(Ft):
            Fc = Ft - Ft.mean(0); fc = fr2 - fr2.mean(); den = np.sqrt((Fc ** 2).sum(0) * (fc ** 2).sum())
            with np.errstate(invalid='ignore', divide='ignore'): cc = np.abs(Fc.T @ fc) / den
            return float(np.nanmax(cc))
        for f_ in fams: cs.append(cmax(hinge(d['pt'], f_, zs)))
        for nm, v in fixed_decoys(d['pt'], n).items(): cs.append(cmax(v))
        R['decoy_max_corr_r2formula'] = cs
    if edits:
        rng = np.random.default_rng(4434)
        u = R['roles']['r2']['unit_idx']
        R['radial_weight'] = radial_weight(model, d, live[u], model.f16, rng, nboot) if u is not None else None
        rw = R['radial_weight'] or {}
        log(f'  w(r) on r2 unit {R["roles"]["r2"]["unit"]}: beta {rw.get("beta")} {rw.get("beta_ci")} p(beta=2) {rw.get("p_beta_eq_2")}; '
            f'azimuthal / radial {rw.get("az_over_radial")} {rw.get("az_over_radial_ci")}; gamma per bin '
            + ', '.join(f'{b["lo"]}-{b["hi"]}: {b.get("gamma", float("nan")):.3g} (n {b["n"]})' for b in rw.get('bins', [])))
        u = R['roles']['hard']['unit_idx']
        R['hard_soft'] = hard_soft(model, d, live[u], model.f16, rng, nboot) if u is not None else None
        log(f'  pT-add on hard unit {R["roles"]["hard"]["unit"]}: ' + json.dumps(E.clean(R['hard_soft'])))
    zw = E.z_windows(model, d['xt'], d['xkt'], np.random.default_rng(4435), nb=50 if per < 2000 else 200)
    R['z_centres'] = sorted(e['centre'] for w in zw['units'].values() for e in w)
    R['z_units'] = zw['units']
    log(f'  z windows: {len(R["z_centres"])} centres ' + ', '.join(f'{u}: {[round(e["centre"]) for e in w]}' for u, w in zw['units'].items()))
    del model; gc.collect()
    try:
        import jax; jax.clear_caches()
    except Exception: pass
    return R


def decoy_fams(ndec):
    rng = np.random.default_rng(4433)
    return [(float(rng.uniform(0, 2 * np.pi)), float(rng.standard_normal())) for _ in range(ndec)]


# ------------------------------------------------------------------------------------------------ combine
def cp(k, n, a=0.05):
    from scipy.stats import beta
    if n == 0: return [None, None]
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1)); hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return [lo, hi]


def combine(tag):
    from scipy.stats import fisher_exact
    def merge(part):
        out = {}
        for f in sorted(OUT.glob(f'e33x_{part}*.json')):
            if ('_smoke' in f.name) != bool(tag): continue
            out.update(json.loads(f.read_text())['nets']); log('combine: read', f.name)
        return out
    nets = merge('trained'); nets = {m: nets[m] for m in E.TRAINED if m in nets}; U = merge('untrained')
    ref = nets['official_n64']; others = [m for m in nets if m != 'official_n64']; un = list(U.values())
    ndec = len(ref['decoys_hung']); val = lambda v: -1.0 if v is None else v  # noqa: E731

    def fval(R, f, mode='hung'):
        if f in ROLES: return val(R['roles'][f]['r2_test'] if mode == 'hung' else R['roles'][f]['best_unassigned'])
        return val(R['competitors'][f]['r2_test'] if mode == 'hung' else R['competitors'][f]['best_unassigned'])
    S = dict(node='E33x', n_decoys=ndec, n_other=len(others), n_untrained=len(un), acc={m: [R['acc'], R['acc_ci']] for m, R in nets.items()})
    for mode, dkey in (('hung', 'decoys_hung'), ('best', 'decoys_best')):
        q = [i for i in range(ndec) if val(ref[dkey][i]) >= 0.9]; nq = len(q)
        drec = np.array([sum(val(nets[m][dkey][i]) >= 0.9 for m in others) for i in q])
        dun = np.array([sum(val(R[dkey][i]) >= 0.9 for R in un) for i in q])
        out = dict(n_qualifying=nq, qualifying_share=nq / ndec, qualifying_share_ci=cp(nq, ndec),
                   decoy_recurrence_hist=np.bincount(drec, minlength=len(others) + 1).tolist() if nq else None,
                   decoy_mean_recurrence=float(drec.mean()) if nq else None, decoy_untrained_mean=float(dun.mean()) if nq else None,
                   formulas={})
        if mode == 'hung' and 'decoy_max_corr_r2formula' in ref:
            cc = np.array(ref['decoy_max_corr_r2formula'])[q]
            out['r2like_split'] = dict(n_r2like=int((cc >= 0.9).sum()), mean_rec_r2like=float(drec[cc >= 0.9].mean()) if (cc >= 0.9).any() else None,
                                       n_other=int((cc < 0.9).sum()), mean_rec_other=float(drec[cc < 0.9].mean()) if (cc < 0.9).any() else None)
        pm = {}
        for f in FORMULAS:
            r0 = fval(ref, f, mode); k = sum(fval(nets[m], f, mode) >= 0.9 for m in others); u = sum(fval(R, f, mode) >= 0.9 for R in un)
            ge = int((drec >= k).sum()) if nq else 0
            e = dict(ref_r2=r0, qualifies=r0 >= 0.9, recurrence=k, per_network={m: fval(nets[m], f, mode) for m in others},
                     share_decoys_ge=ge / nq if nq else None, share_ci=cp(ge, nq), p=(1 + ge) / (1 + nq) if nq else None,
                     untrained_count=u, untrained_rate=u / len(un), untrained_ci=cp(u, len(un)),
                     p_fisher_trained_gt_untrained=float(fisher_exact([[k, len(others) - k], [u, len(un) - u]], alternative='greater')[1]))
            if mode == 'hung':
                for m in others:
                    v = fval(nets[m], f, mode); dv = np.array([val(nets[m][dkey][i]) for i in q])
                    pm[(f, m)] = (1 + int((dv >= v).sum())) / (1 + nq) if nq else None
            out['formulas'][f] = e
        if mode == 'hung':
            keys = list(pm); hp = E.holm([pm[k] for k in keys])
            for (f, m), p, h in zip(keys, [pm[k] for k in keys], hp):
                out['formulas'][f].setdefault('per_network_p', {})[m] = dict(p=p, p_holm=h)
            pf = [out['formulas'][f]['p'] for f in FORMULAS]; hf = E.holm(pf)
            for f, h in zip(FORMULAS, hf): out['formulas'][f]['p_holm_formulas'] = h
        S[mode] = out
        log(f'[{mode}] qualifying decoys {nq}/{ndec}; decoy recurrence mean {out["decoy_mean_recurrence"]} hist {out["decoy_recurrence_hist"]}; untrained decoy mean {out["decoy_untrained_mean"]}')
        for f, e in out['formulas'].items():
            log(f'   {f}: ref R2 {e["ref_r2"]:.3f} recurrence {e["recurrence"]}/9, share decoys >= {e["share_decoys_ge"]} {e["share_ci"]} p {e["p"]} '
                f'(Holm {e.get("p_holm_formulas")}); untrained {e["untrained_count"]}/{len(un)} Fisher p {e["p_fisher_trained_gt_untrained"]:.3g}')
    # radial weight / pT-add
    S['radial_weight'] = {m: {k: v for k, v in (R.get('radial_weight') or {}).items()} for m, R in nets.items()}
    S['hard_soft'] = {m: R.get('hard_soft') for m, R in nets.items()}
    S['roles_units'] = {m: {r: (R['roles'][r]['unit'], R['roles'][r]['param'], R['roles'][r]['r2_test']) for r in ROLES} for m, R in nets.items()}
    # z windows positional null
    W = ref['z_centres']; C = {m: np.array(nets[m]['z_centres']) for m in others}

    def stat(Cs):
        return sum(int(any(abs(c - w) <= 10 for c in cc)) for w in W for cc in Cs)
    Tobs = stat(list(C.values())); rng = np.random.default_rng(4436); Tn = []
    for _ in range(1000): Tn.append(stat([cc + rng.uniform(-50, 50) for cc in C.values()]))
    Tn = np.array(Tn); Tu = stat([np.array(R['z_centres']) for R in un])
    S['z_windows'] = dict(ref_centres=W, centres={m: nets[m]['z_centres'] for m in others}, T=Tobs, max_T=len(W) * len(others),
                          null_mean=float(Tn.mean()), null_q=[float(np.quantile(Tn, q)) for q in (.025, .5, .975)],
                          p=float((1 + (Tn >= Tobs).sum()) / 1001), T_untrained=Tu, T_untrained_scaled9=Tu * len(others) / max(len(un), 1),
                          untrained_centres={f'{R["model"]}_u{R["untrained_seed"]}': R['z_centres'] for R in un})
    log(f'z windows: ref {np.round(W).tolist()}; T {Tobs}/{len(W) * len(others)} vs shifted {Tn.mean():.2f} [{np.quantile(Tn, .025)}, {np.quantile(Tn, .975)}] p {S["z_windows"]["p"]:.4f}; untrained T {Tu} over {len(un)} nets')
    for m, rw in S['radial_weight'].items():
        log(f'w(r) {m}: beta {rw.get("beta")} {rw.get("beta_ci")} az/radial {rw.get("az_over_radial")}; pT-add soft/hard {(S["hard_soft"][m] or {}).get("ratio")} {(S["hard_soft"][m] or {}).get("ratio_ci")}')
    (OUT / f'e33x{tag}.json').write_text(json.dumps(E.clean(S), indent=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--part', required=True, choices=['trained', 'untrained', 'combine'])
    ap.add_argument('--models', default=None, help='comma list (default: all trained)'); ap.add_argument('--label', default='')
    ap.add_argument('--smoke', action='store_true'); a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    tag = '_smoke' if a.smoke else ''
    if a.part == 'combine': combine(tag); return
    per, ndec, nboot = (200, 20, 50) if a.smoke else (2000, 600, 1000)
    fams = decoy_fams(ndec); zs = zstats(per)
    models = a.models.split(',') if a.models else (E.TRAINED if not a.smoke else ['official_n64', 'hgq150_s0', 'f16_n64'])
    jobs = [(m, None) for m in models] if a.part == 'trained' else [(m, s) for s in (SEEDS_U if not a.smoke else (0,)) for m in models]
    S = dict(node='E33x', part=a.part, samples=f'fit sample({per}, 4401), test sample({per}, 4402)', decoy_families=fams, z_standardisation=zs, nets={})
    JS = OUT / f'e33x_{a.part}{"_" + a.label if a.label else ""}{tag}.json'
    for m, s in jobs:
        key = m if s is None else f'{m}_u{s}'
        log('network', key); S['nets'][key] = run_net(m, s, per, fams, zs, nboot, edits=(a.part == 'trained'))
        JS.write_text(json.dumps(E.clean(S), indent=1))
    log('done', JS)


if __name__ == '__main__':
    main()
