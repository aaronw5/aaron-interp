"""E33x verification (fresh jets + matched-complexity decoy control).  No training.

  python -m interp.e33x_verify [--smoke]   -> $AI_RESULTS/jedi/e33x_verify[_smoke].json

1. Replication on fresh jets: fit = jedi_core.sample(2000, 5501), test = sample(2000, 5502) (10,000 each; E33x used 4401/4402).
   E33x's fits() is rerun unchanged on all 10 trained networks (4 roles by Hungarian matching, 4 competitors, the same 603
   single-column rank-window hinge decoys, rng 4433).  Recurrence of r2 and the decoy p as in E33x (hung and unassigned).
2. Matched-complexity control (missing in E33x): the r2 formula is a linear fit on 3 regressors (sum pT r^2, sum r^2,
   N(pT < 2, r > 0.1)); the E33x decoys are fitted with 1 regressor.  Two 3-regressor decoy sets, each fitted to every live S unit
   by least squares on [1, F1, F2, F3] exactly as the r2 formula:
     D3h: 300 triples of random per-particle hinges summed over all real particles,
          F = sum_i ReLU(cos th z_pT,i + sin th z_r,i + b), th ~ U(0, 2 pi), b ~ N(0, 1) (rng 4477);
     D3L: 300 random triples of the per-particle feature library sums over all particles (E33 library '[all]' columns:
          count, pT, ln pT, pT^2, r, r^2, pT r, pT r^2, ln pT r, soft-wide count, (r - t)+, (t - r)+, 1[pT < t], (pT - t)+,
          (t - pT)+, 1[r > t], ln pT (r - t)+), the exact r2 triple excluded (rng 4478).
   Each is scored both unassigned (best unit by fit R^2) and as a 5th row in the Hungarian matching with the 4 roles (as E33x).
   p = (1 + #{qualifying D3 decoys with recurrence >= r2's}) / (1 + #qualifying).
3. r2 ablations (3-regressor or fewer, same fit): [sum pT r^2] alone; [sum pT r^2, sum r^2]; [sum r^2, N_soft] (no pT
   weighting); [sum pT r, sum r, N_soft] (the beta = 1 analogue); [sum pT r^3, sum r^3, N_soft] (beta = 3 analogue).
4. Radial weight beta on each trained network's r2-assigned unit on the fresh test jets (E33x radial_weight, 500 draws).
5. Untrained float redraws (float_s0, float_s1; seeds 0-2): r2 test R^2 (Hungarian and unassigned).
"""
import argparse, contextlib, gc, json, os, sys, time
from pathlib import Path
import numpy as np

os.environ.setdefault('KERAS_BACKEND', 'jax')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import e33_program_recurrence_jedi as E  # noqa: E402
from interp import e33x_recurrence_null as X  # noqa: E402

OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
T0 = time.time()
SF, ST = 5501, 5502


def log(*a): print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


_D = {}


def data(per, n, f16):
    key = (n, f16)
    if key not in _D:
        if n == 128: _D.pop((128, False), None)
        xf, xkf, x3f, yf, _ = E.load(per, SF, n, f16); xt, xkt, x3t, yt, _ = E.load(per, ST, n, f16)
        _D[key] = dict(xf=xf, xkf=xkf, yf=yf, xt=xt, xkt=xkt, yt=yt, pf=E.particles(xkf), pt=E.particles(xkt))
    return _D[key]


def lib_all(xk):
    L, names = E.library(xk); keep = [i for i, nm in enumerate(names) if nm.endswith('[all]')]
    return L[:, keep], [names[i] for i in keep]


def hinge_all(P, fam, zs):
    pt, r, real, _ = P; th, b = fam
    zp = (pt - zs[0]) / zs[1]; zr = (r - zs[2]) / zs[3]
    return (real * np.maximum(np.cos(th) * zp + np.sin(th) * zr + b, 0)).sum(1)


def ablations(P):
    pt, r, real, _ = P; rf = real.astype(float); ns = (rf * ((pt < 2) & (r > 0.1))).sum(1)
    return dict(pTr2=np.stack([(pt * r ** 2).sum(1)], 1), pTr2_r2=np.stack([(pt * r ** 2).sum(1), (rf * r ** 2).sum(1)], 1),
                r2_Nsoft=np.stack([(rf * r ** 2).sum(1), ns], 1), beta1=np.stack([(pt * r).sum(1), (rf * r).sum(1), ns], 1),
                beta3=np.stack([(pt * r ** 3).sum(1), (rf * r ** 3).sum(1), ns], 1))


def role_rows(Zf, Zt, xkf, xkt, nlive):
    RF, RT = E.role_feats(xkf), E.role_feats(xkt); rf, rt = [], []
    for role in E.ROLES:
        if not nlive: rf.append(np.zeros(0)); rt.append(np.zeros(0)); continue
        if role == 'r2':
            f, t, _ = E.fit_set(Zf, Zt, RF['r2'][0][1], RT['r2'][0][1])
        else:
            Ff = np.concatenate([g for _, g in RF[role]], 1); Ft = np.concatenate([g for _, g in RT[role]], 1)
            f, t, _ = X.fam_unit(Zf, Zt, Ff, Ft)
        rf.append(f); rt.append(t)
    return rf, rt


def multi_row(Zf, Zt, Gf, Gt, rf, rt):
    """3-regressor formula as a 5th Hungarian row and unassigned -> (hung test R2, unassigned test R2)"""
    if Zf.shape[1] == 0: return None, None
    f, t, _ = E.fit_set(Zf, Zt, Gf, Gt)
    return X.hung(rf + [f], rt + [t])[-1][1], X.unassigned(f, t)


def one(name, seed, per, fams, zs, d3h, d3l, full, nboot):
    kind, fam, n = E.MODELS[name]
    ctx = X.redrawn(seed) if seed is not None else contextlib.nullcontext()
    with ctx:
        model = E.Model(name)
    d = data(per, n, model.f16)
    if full:
        R, Sf, St, live = X.fits(model, d, n, fams, zs)
    else:
        St_ = model.run(d['xt'], keep=('S', 'logits')); Sf = model.run(d['xf'], keep=('S',))['S']; St = St_['S']
        live = [c for c in range(64) if Sf[:, c].std() > 1e-9]
        R = dict(acc=float((St_['logits'].argmax(1) == d['yt']).mean()), n_live_S=len(live))
    Zf, Zt = Sf[:, live], St[:, live]
    rf, rt = role_rows(Zf, Zt, d['xkf'], d['xkt'], len(live))
    asg = X.hung(rf, rt) if len(live) else [(None, None)] * 4
    R['r2_hung'] = asg[1][1]; R['r2_unassigned'] = X.unassigned(rf[1], rt[1]) if len(live) else None
    R['r2_unit'] = None if asg[1][0] is None else f'S{live[asg[1][0]]}'
    R['model'] = name; R['seed'] = seed
    if full:
        R['ablations'] = {}
        Af, At = ablations(d['pf']), ablations(d['pt'])
        for k in Af: R['ablations'][k] = multi_row(Zf, Zt, Af[k], At[k], rf, rt)
        Lf, _ = lib_all(d['xkf']); Lt, _ = lib_all(d['xkt'])
        R['d3h'] = [multi_row(Zf, Zt, np.stack([hinge_all(d['pf'], f_, zs) for f_ in tr], 1),
                              np.stack([hinge_all(d['pt'], f_, zs) for f_ in tr], 1), rf, rt) for tr in d3h]
        R['d3l'] = [multi_row(Zf, Zt, Lf[:, list(tr)], Lt[:, list(tr)], rf, rt) for tr in d3l]
        u = asg[1][0]
        R['radial_weight'] = X.radial_weight(model, d, live[u], model.f16, np.random.default_rng(4434), nboot) if u is not None else None
        rw = R['radial_weight'] or {}
        log(f'  beta {rw.get("beta")} {rw.get("beta_ci")} az/rad {rw.get("az_over_radial")}; gamma '
            + ', '.join(f'{b.get("gamma", float("nan")):.3g}' for b in rw.get('bins', [])))
    log(f'{name} seed {seed}: acc {R["acc"]:.4f} live {R["n_live_S"]} r2 {R["r2_unit"]} hung {R["r2_hung"]} unassigned {R["r2_unassigned"]}'
        + (f'; abl {R["ablations"]}' if full else ''))
    del model; gc.collect()
    try:
        import jax; jax.clear_caches()
    except Exception: pass
    return R


def recur(ref, others, get, thr=0.9):
    v = lambda x: -1.0 if x is None else x  # noqa: E731
    q = [i for i in range(len(get(ref))) if v(get(ref)[i]) >= thr]
    rec = np.array([sum(v(get(o)[i]) >= thr for o in others) for i in q])
    return q, rec


def summarise(nets):
    from scipy.stats import fisher_exact  # noqa: F401
    ref = nets['official_n64']; others = [nets[m] for m in E.TRAINED if m != 'official_n64' and m in nets]
    v = lambda x: -1.0 if x is None else x  # noqa: E731
    S = {}
    for thr in (0.85, 0.9, 0.95):
        T = {}
        for mode, ix in (('hung', 0), ('unassigned', 1)):
            r2key = 'r2_hung' if mode == 'hung' else 'r2_unassigned'
            k = sum(v(o[r2key]) >= thr for o in others); r2ref = v(ref[r2key])
            T[mode] = dict(r2_ref=r2ref, r2_recurrence=k, r2_per_network={o['model']: o[r2key] for o in others})
            for nm, get in (('E33x_decoys', lambda R: R['decoys_hung' if mode == 'hung' else 'decoys_best']),
                            ('D3h', lambda R: [t[ix] for t in R['d3h']]), ('D3L', lambda R: [t[ix] for t in R['d3l']])):
                q, rec = recur(ref, others, get, thr); ge = int((rec >= k).sum()) if len(q) else 0
                T[mode][nm] = dict(n=len(get(ref)), n_qualifying=len(q), mean_recurrence=float(rec.mean()) if len(q) else None,
                                   hist=np.bincount(rec, minlength=10).tolist() if len(q) else None, n_ge_r2=ge,
                                   share_ge=ge / len(q) if len(q) else None, share_ci=X.cp(ge, len(q)), p=(1 + ge) / (1 + len(q)))
            T[mode]['ablations'] = {a: dict(ref=ref['ablations'][a][ix], recurrence=sum(v(o['ablations'][a][ix]) >= thr for o in others),
                                            per_network={o['model']: o['ablations'][a][ix] for o in others}) for a in ref['ablations']}
        S[f'thr_{thr}'] = T
    return S


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    per, ndec, n3, nboot = (200, 20, 20, 30) if a.smoke else (2000, 600, 300, 500)
    fams = X.decoy_fams(ndec)
    d = data(per, 64, False); p, r, re_, _ = d['pf']; zs = (float(p[re_].mean()), float(p[re_].std()), float(r[re_].mean()), float(r[re_].std()))
    rng = np.random.default_rng(4477)
    d3h = [[(float(rng.uniform(0, 2 * np.pi)), float(rng.standard_normal())) for _ in range(3)] for _ in range(n3)]
    _, names = lib_all(d['xkf'][:10]); excl = {names.index('pTr2[all]'), names.index('r2[all]'), names.index('soft_wide[all]')}
    rng = np.random.default_rng(4478); d3l = []
    while len(d3l) < n3:
        tr = tuple(sorted(rng.choice(len(names), 3, replace=False).tolist()))
        if set(tr) != excl and tr not in d3l: d3l.append(tr)
    models = E.TRAINED if not a.smoke else ['official_n64', 'hgq150_s0', 'f16_n64']
    tag = '_smoke' if a.smoke else ''; JS = OUT / f'e33x_verify{tag}.json'
    S = dict(node='E33x-verify', samples=f'fit sample({per}, {SF}), test sample({per}, {ST})', z_standardisation=zs, d3h_families=d3h,
             d3l_triples=[[names[i] for i in tr] for tr in d3l], library_names=names, nets={}, untrained={})
    for m in models:
        log('network', m); S['nets'][m] = one(m, None, per, fams, zs, d3h, d3l, True, nboot)
        JS.write_text(json.dumps(E.clean(S), indent=1))
    S['summary'] = summarise(S['nets'])
    for thr, T in S['summary'].items():
        for mode, U in T.items():
            log(f'{thr} {mode}: r2 ref {U["r2_ref"]:.3f} rec {U["r2_recurrence"]}/9; '
                + '; '.join(f'{nm}: q {U[nm]["n_qualifying"]}/{U[nm]["n"]} mean {U[nm]["mean_recurrence"]} >=r2 {U[nm]["n_ge_r2"]} p {U[nm]["p"]:.4f}'
                            for nm in ('E33x_decoys', 'D3h', 'D3L'))
                + '; abl ' + ', '.join(f'{k} {v["ref"]} rec {v["recurrence"]}' for k, v in U['ablations'].items()))
    JS.write_text(json.dumps(E.clean(S), indent=1))
    for m in (['float_s0', 'float_s1'] if not a.smoke else ['float_s0']):
        for s in ((0, 1, 2) if not a.smoke else (0,)):
            S['untrained'][f'{m}_u{s}'] = one(m, s, per, fams, zs, d3h, d3l, False, nboot)
            JS.write_text(json.dumps(E.clean(S), indent=1))
    log('done', JS)


if __name__ == '__main__':
    main()
