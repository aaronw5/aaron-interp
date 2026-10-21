"""H69 verification (loop D): rerun the key H69 edits on a fresh sample and add the missing pT-reweighting controls.

Why: H69's control crot_dm matches the particle *displacement* of the shape edits, but the shape edits mainly *reweight* particle pT
(moved pT share 2-6 % of the jet pT, against ~1 % for crot_dm and ~0 for crand). A network that reacts to any pT reweighting would give
R > 1 without reading s, D2 or tau21. Controls added here (each matched per jet to the mean moved pT share of that jet's converged
shape edits, Es+-10, ED2+-10, Et+-10):
  crw     random within-prong pT reweighting pT_i x (1 + a h_i), h = iid normal per particle made pT-weighted-orthogonal inside each prong
          to {1, d_eta, d_phi, r^2} (prong pT, centroid and sum pT r^2 held, as for H69's width lever), max|h| = 1 per prong; then
          m, w, s and D2 restored exactly (levers f, lu, c, cs). tau21 free (reported).
  crw_sm  as crw, but h = a random smooth profile (random combination of 2nd/3rd-order polynomials of the particle's offset from its
          prong centroid), same orthogonalisation and restore.
  crw_ms  as crw (iid), only m and s restored (levers f, lu); w, D2, tau21 free (reported).
Also: Ew+-10cs = H69's width edit with the prong-mass lever cs added (levers f, lu, c, cs; targets m, w, s): does width become reachable
when the prongs may change their mass? And the spread (5-95 %) of w / (sqrt(2 s) m / pT) on unedited two-prong signal jets.
Samples (fresh): JetClass val_5M <sample>_121.root (H69 used _122), 500 random rows per class of Wqq, Zqq, Tbqq, Hbb, QCD (rng 6990);
  Top Landscape test, 1,000 + 1,000 random rows excluding H69's rows (rng 6991).
Run
  AI_PY=conda ./job.sh d-h69-verify 2 6 'python -m interp.h69v_verify --part edits --ds jetclass && python -m interp.h69v_verify --part edits --ds top'
  ./part_job.sh d-h69-verify-gpu 'python -m interp.h69v_verify --part eval --ds jetclass && python -m interp.h69v_verify --part eval --ds top'
  AI_PY=conda ./job.sh d-h69-verify-sum 'python -m interp.h69v_verify --part summary'
  -> $AI_RESULTS/physics/h69_verify_work/*, h69_verify_edits_<ds>.json, h69_verify.json
"""
import argparse
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import h69_width_sharing_d2 as X  # noqa: E402
from interp import h67_mass_vs_width as M  # noqa: E402
from interp import h12_boost_panel as H  # noqa: E402

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results'))
SMOKE = int(os.environ.get('H69V_SMOKE', '0'))
OUT = RES / 'physics'
WORK = OUT / ('h69_verify_work_smoke' if SMOKE else 'h69_verify_work')
X.WORK = WORK; M.WORK = WORK
SEEDS = dict(jetclass=6990, top=6991)
SHAPE = ['Es+10', 'Es-10', 'ED2+10', 'ED2-10', 'Et+10', 'Et-10']
CTRL = ['crot_dm', 'crw', 'crw_sm', 'crw_ms', 'crand']
EDITS = SHAPE + ['Ew+10cs', 'Ew-10cs'] + CTRL
X.EDITS = EDITS
L = math.log
SPEC = dict(X.SPEC)
SPEC['Ew+10cs'] = (('f', 'lu', 'c', 'cs'), (('m', 0.0), ('w', L(1.1)), ('s', 0.0)))
SPEC['Ew-10cs'] = (('f', 'lu', 'c', 'cs'), (('m', 0.0), ('w', L(0.9)), ('s', 0.0)))
SPEC['crw'] = (('f', 'lu', 'c', 'cs'), (('m', 0.0), ('w', 0.0), ('s', 0.0), ('D2', 0.0)))
SPEC['crw_sm'] = SPEC['crw']
SPEC['crw_ms'] = (('f', 'lu'), (('m', 0.0), ('s', 0.0)))
NB = 1000
T00 = time.time()


def log(*a):
    print(f'[{time.time() - T00:7.0f}s]', *a, flush=True)


gather, prong_sum, prong_centroid = M.gather, M.prong_sum, M.prong_centroid


def hgen(G, pt0, x, v):
    """H69.hfun with an arbitrary target vector v in place of r: v made pT-weighted-orthogonal inside each prong to
    {1, x_eta, x_phi, r^2}, scaled to max |h| = 1 per prong (prongs with < 5 particles get h = 0)"""
    real, lab, K = G['real'], G['lab'], G['K']; r = np.hypot(x[..., 0], x[..., 1])
    c = gather(prong_centroid(pt0, x, lab, K), lab); d = x - c
    sig = np.sqrt(np.maximum(gather(prong_sum(pt0 * (d ** 2).sum(-1), lab, K) / np.maximum(prong_sum(pt0, lab, K), 1e-300), lab), 1e-12))
    Pk = np.maximum(prong_sum(pt0, lab, K), 1e-300)
    r2c = gather(prong_sum(pt0 * r ** 2, lab, K) / Pk, lab); r1c = gather(prong_sum(pt0 * r, lab, K) / Pk, lab)
    B = np.stack([np.ones_like(r), d[..., 0] / sig, d[..., 1] / sig, (r ** 2 - r2c) / np.maximum(sig * r1c, 1e-12)], -1)
    B = np.where(real[..., None], B, 0)
    Wk = np.stack([np.where(real & (lab == k), pt0, 0) for k in range(K)], -1)
    Gm = np.einsum('jlk,jla,jlb->jkab', Wk, B, B); rhs = np.einsum('jlk,jla,jl->jka', Wk, B, v)
    tr = np.trace(Gm, axis1=2, axis2=3)[..., None, None]
    coef = np.linalg.solve(Gm + (1e-10 * tr + 1e-30) * np.eye(4), rhs[..., None])[..., 0]
    h = np.where(real, v - np.einsum('jla,jla->jl', B, gather(coef, lab)), 0)
    nk = prong_sum(real.astype(float), lab, K)
    mx = np.stack([np.where(lab == k, np.abs(h), 0).max(1) for k in range(K)], 1)
    ok = (nk >= 5) & (mx > 0)
    h = np.where(gather(ok, lab), h / np.maximum(gather(mx, lab), 1e-300), 0)
    return np.where(real, h, 0), d / sig[..., None]


def make_edits(T0, y, ntop, seed):
    G = X.geometry69(T0, ntop); J = len(y); pt = T0['pt']
    out = {}
    for e in SHAPE + ['Ew+10cs', 'Ew-10cs']:
        t = time.time(); pr, tg = SPEC[e]; U, inf = X.solve3(T0, G, pr, tg); out[e] = (U, inf); c = inf['conv']
        log(f'{e}: converged {c.mean():.3f} (' + ' '.join(f'{k}={c[y == k].mean():.2f}' for k in sorted(set(y.tolist()))) + f') ({time.time() - t:.0f} s)')
    # matched sizes: mean over this jet's converged 10 % shape edits (displacement for crot_dm, moved pT share for crw*)
    Dm = np.stack([np.where(out[e][1]['conv'], M.displacement(G, out[e][0], pt), np.nan) for e in SHAPE], 1)
    Sm = np.stack([np.where(out[e][1]['conv'], M.moved_share(T0, out[e][0]), np.nan) for e in SHAPE], 1)
    with np.errstate(all='ignore'): D = np.nanmean(Dm, 1); S = np.nanmean(Sm, 1)
    D = np.where(np.isfinite(D), D, np.nanmedian(D)); S = np.where(np.isfinite(S), S, np.nanmedian(S))
    rng = np.random.default_rng(seed + 2000)
    rbar = (pt * G['rho']).sum(1) / np.maximum(pt.sum(1), 1e-300)
    th = 2 * np.arcsin(np.clip(D / np.maximum(2 * rbar, 1e-12), 0, 1)) * rng.choice([-1.0, 1.0], J)
    U, inf = X.solve3(T0, G, *SPEC['crot_dm'], base_x=X.rotate_prongs(G, th)); out['crot_dm'] = (U, inf)
    log(f'crot_dm: converged {inf["conv"].mean():.3f}')
    # random reweightings
    v_iid = np.where(G['real'], rng.standard_normal(pt.shape), 0)
    _, u = hgen(G, pt, G['x0'], np.zeros_like(pt))
    a0, a1 = u[..., 0], u[..., 1]
    terms = np.stack([a0 ** 2 - a1 ** 2, a0 * a1, a0 ** 3, a1 ** 3, a0 ** 2 * a1, a0 * a1 ** 2], -1)
    cf = rng.standard_normal((J, G['K'], terms.shape[-1]))
    v_sm = np.where(G['real'], np.einsum('jlt,jlt->jl', terms, gather(cf, G['lab'])), 0)
    for nm, v in (('crw', v_iid), ('crw_sm', v_sm), ('crw_ms', v_iid)):
        h, _ = hgen(G, pt, G['x0'], v)
        den = (pt * np.abs(h)).sum(1); a = np.clip(S * 2 * pt.sum(1) / np.maximum(den, 1e-300), 0, 0.9)
        base = np.where(G['real'], pt * (1 + a[:, None] * h), 0)
        t = time.time(); U, inf = X.solve3(T0, G, *SPEC[nm], base_pt=base); out[nm] = (U, inf)
        log(f'{nm}: converged {inf["conv"].mean():.3f}; moved share median {np.median(M.moved_share(T0, U)[inf["conv"]]):.4f} vs matched {np.median(S):.4f} '
            f'(a capped for {(a >= 0.9).mean():.3f}) ({time.time() - t:.0f} s)')
    a = rng.uniform(0, 2 * np.pi, pt.shape); xr = np.where(G['real'][..., None], G['x0'] + D[:, None, None] * np.stack([np.cos(a), np.sin(a)], -1), 0)
    U, inf = X.solve3(T0, G, *SPEC['crand'], base_x=xr); out['crand'] = (U, inf)
    return out, G


def part_edits(ds):
    seed = SEEDS[ds]; t0 = time.time()
    if ds == 'jetclass':
        from interp.h1_irc_panel import jc_table
        from interp.part_core import cat, CL, PRE
        rng = np.random.default_rng(seed)
        J0 = cat([X.jc_rows(X.JCV / 'val_5M' / f'{PRE[c]}_121.root', 500, rng) for c in X.JC_CL])
        Tj = jc_table(J0); T0 = dict(pt=Tj['pt'], eta=Tj['eta'], phi=Tj['phi'], m=Tj['m'], ax_eta=Tj['ax_eta'], ax_phi=Tj['ax_phi'], ext=Tj['ext'])
        y = J0['label'].astype(int); classes = list(CL); ntop = y == CL.index('Tbqq')
        info = f'JetClass val_5M <sample>_121.root, 500 random rows per class of {X.JC_CL} (rng {seed})'
    else:
        from interp import data_top as dt
        d = dt.DST / 'test'; ylab = np.load(d / 'label.npy'); old = set(H._rows(ylab, 2000, X.SEEDS['top']).tolist())
        rng = np.random.default_rng(seed)
        ev = np.sort(np.concatenate([rng.choice(np.array([i for i in np.flatnonzero(ylab == c) if i not in old]), 1000, replace=False) for c in (0, 1)]))
        Fm = np.load(d / 'full.npy', mmap_mode='r'); jet = np.load(d / 'jet.npy', mmap_mode='r')
        T0 = H.table_from_full(np.asarray(Fm[ev]), np.minimum(np.asarray(jet[ev, 5]).astype(np.int64), Fm.shape[1])); y = ylab[ev].astype(int)
        classes = ['QCD', 'top']; ntop = y == 1; J0 = None
        info = f'Top Landscape test: 1,000 top + 1,000 QCD random rows not in the H69 sample (rng {seed})'
    if SMOKE:
        rs = np.random.default_rng(1); ii = np.sort(np.concatenate([rs.choice(np.flatnonzero(y == c), SMOKE, replace=False) for c in sorted(set(y.tolist()))]))
        T0 = {k: (v[ii] if isinstance(v, np.ndarray) else ({a: b[ii] for a, b in v.items()} if k == 'ext' else v)) for k, v in T0.items()}
        y, ntop = y[ii], ntop[ii]
        if J0 is not None:
            from interp.part_core import sub
            J0 = sub(J0, ii)
    T0 = M.sort_first(T0)
    log(f'{ds}: {len(y)} jets {np.bincount(y).tolist()} ({time.time() - t0:.0f} s)')
    out, G = make_edits(T0, y, ntop, seed)
    o0 = G['o0']; _, _, _, m0 = X.jet3(T0['pt'], T0['eta'], T0['phi'], T0['m'])
    pred = np.sqrt(2 * np.exp(o0['s'])) * m0 / G['jpt0']; rw = np.exp(o0['w']) / np.maximum(pred, 1e-12)
    res = dict(sample=info, N=int(len(y)), identity={}, edits={})
    for k in sorted(set(y.tolist())):
        sel = (y == k) & ~ntop
        if sel.sum() > 5: res['identity'][classes[k]] = dict(median=float(np.median(rw[sel])), q5_95=np.percentile(rw[sel], [5, 95]).tolist(),
                                                             q25_75=np.percentile(rw[sel], [25, 75]).tolist())
    per = dict(y=y.astype(np.int16), ntop=ntop)
    for e, (U, inf) in out.items():
        c = inf['conv']; o1 = X.observe(U, G, ('m', 'w', 's', 'D2', 'tau21')); dd = {q: o1[q] - o0[q] for q in o1}
        jpt, je, jp, _ = X.jet3(U['pt'], U['eta'], U['phi'], U['m'])
        mv = M.moved_share(T0, U); disp = M.displacement(G, U, T0['pt'])
        per[f'{e}__conv'] = c; per[f'{e}__moved'] = mv; per[f'{e}__disp'] = disp
        for q in dd: per[f'{e}__dlog{q}'] = dd[q]
        er = dict(conv_frac=float(c.mean()), conv_by_class={classes[k]: float(c[y == k].mean()) for k in sorted(set(y.tolist()))},
                  moved_median=float(np.median(mv[c])) if c.any() else None, disp_median=float(np.median(disp[c])) if c.any() else None,
                  max_abs_dlogm=float(np.abs(dd['m'][c]).max()) if c.any() else None, max_abs_dlogpt=float(np.abs(np.log(jpt / G['jpt0']))[c].max()) if c.any() else None,
                  change={q: {classes[k]: dict(median=float(np.median(dd[q][c & (y == k)])), q5_95=np.percentile(dd[q][c & (y == k)], [5, 95]).tolist())
                              for k in sorted(set(y.tolist())) if (c & (y == k)).sum() > 0} for q in dd})
        res['edits'][e] = er
        log(f'{e}: conv {c.mean():.3f} moved {er["moved_median"]} disp {er["disp_median"]} | med dlog w {np.median(dd["w"][c]) if c.any() else 0:+.4f} '
            f's {np.median(dd["s"][c]) if c.any() else 0:+.4f} D2 {np.median(dd["D2"][c]) if c.any() else 0:+.4f} tau21 {np.median(dd["tau21"][c]) if c.any() else 0:+.4f} '
            f'|tau21| {np.median(np.abs(dd["tau21"][c])) if c.any() else 0:.4f}')
    WORK.mkdir(parents=True, exist_ok=True)
    (OUT / f'h69_verify_edits_{ds}{"_smoke" if SMOKE else ""}.json').write_text(json.dumps(res, indent=1, default=float))
    D = dict(y=y.astype(np.int16), ntop=ntop)
    for nm, T in [('U', T0)] + [(e, out[e][0]) for e in EDITS]:
        for k in ('pt', 'eta', 'phi', 'm'): D[f'{nm}__{k}'] = T[k].astype(np.float64 if k in ('eta', 'phi') else np.float32)
        D[f'{nm}__ax_eta'] = T['ax_eta']; D[f'{nm}__ax_phi'] = T['ax_phi']
    np.savez_compressed(WORK / f'{ds}_tables.npz', **D)
    np.savez_compressed(WORK / f'{ds}_perjet.npz', **per)
    if J0 is not None:
        with open(WORK / f'{ds}_extra.pkl', 'wb') as fh: pickle.dump(dict(J0=J0, ext=T0['ext']), fh, protocol=4)
    log(f'wrote {ds} tables')


def sflip_p(d):
    d = np.asarray(d, float); n = len(d)
    if n < 2: return 1.0
    return float(math.erfc(abs(d.mean()) / max(math.sqrt((d ** 2).sum()) / n, 1e-300) / math.sqrt(2)))


def part_summary():
    rng = np.random.default_rng(6999); S = dict(datasets={}, holm={})
    tests = []
    for ds, cls_eval in (('jetclass', ['Wqq', 'Zqq', 'Hbb', 'Tbqq', 'QCD']), ('top', ['top', 'QCD'])):
        if not (WORK / f'{ds}_logits.npz').exists(): log(f'{ds}: no logits'); continue
        P = np.load(WORK / f'{ds}_perjet.npz'); y = P['y'].astype(int)
        if ds == 'jetclass':
            from interp.part_core import CL
            classes = list(CL)
        else: classes = ['QCD', 'top']
        z = np.load(WORK / f'{ds}_logits.npz'); Lg = {}
        for k in z.files: mn, e = k.split('__'); Lg.setdefault(mn, {})[e] = z[k]
        D = dict(edits=json.loads((OUT / f'h69_verify_edits_{ds}.json').read_text()), models={})
        for mn, lg in Lg.items():
            lo = {e: M.logodds_all(v) for e, v in lg.items()}; c0 = lg['U'].argmax(1)
            r = dict(acc_by_class={cl: float((c0[y == classes.index(cl)] == classes.index(cl)).mean()) for cl in cls_eval},
                     acc=float((c0 == y).mean()), cls={})
            for cl in cls_eval:
                k = classes.index(cl); sel = y == k
                Ds = {e: (lo[e][:, k] - lo['U'][:, k])[sel] for e in EDITS}
                cv = {e: P[f'{e}__conv'][sel] for e in EDITS}; mv = {e: P[f'{e}__moved'][sel] for e in EDITS}
                rc = dict(mean_abs={}, R={}, per_moved={})
                for e in EDITS:
                    if cv[e].sum() >= 10:
                        A = np.abs(Ds[e][cv[e]]); I = rng.integers(0, len(A), (NB, len(A)))
                        rc['mean_abs'][e] = dict(n=int(cv[e].sum()), v=float(A.mean()), ci=np.percentile(A[I].mean(1), [2.5, 97.5]).tolist(),
                                                 moved=float(mv[e][cv[e]].mean()))
                        rc['per_moved'][e] = float(A.mean() / max(mv[e][cv[e]].mean(), 1e-12))
                for e in SHAPE:
                    for C in ('crot_dm', 'crw', 'crw_sm', 'crw_ms'):
                        sl = cv[e] & cv[C]; n = int(sl.sum())
                        if n < 15: continue
                        A, B = np.abs(Ds[e][sl]), np.abs(Ds[C][sl]); I = rng.integers(0, n, (NB, n))
                        with np.errstate(all='ignore'): Rb = A[I].mean(1) / B[I].mean(1)
                        p = sflip_p(A - B)
                        rc['R'][f'{e}|{C}'] = dict(n=n, R=float(A.mean() / max(B.mean(), 1e-300)), ci=np.nanpercentile(Rb, [2.5, 97.5]).tolist(), p=p,
                                                   moved_edit=float(mv[e][sl].mean()), moved_ctrl=float(mv[C][sl].mean()))
                        if 'perm' not in mn and C == 'crw': tests.append((f'{ds}|{mn}|{cl}|{e}|{C}', p))
                # grouped (both signs) R vs each control, and phi_sD2
                g = {}
                for Xn, es in (('s', ('Es+10', 'Es-10')), ('D2', ('ED2+10', 'ED2-10')), ('tau21', ('Et+10', 'Et-10'))):
                    vals = [rc['mean_abs'][e]['v'] for e in es if e in rc['mean_abs']]
                    if vals: g[Xn] = float(np.mean(vals))
                rc['M'] = g
                for C in ('crot_dm', 'crw', 'crw_sm', 'crw_ms', 'crand'):
                    if C in rc['mean_abs']: rc[f'Rgroup_{C}'] = {Xn: v / max(rc['mean_abs'][C]['v'], 1e-300) for Xn, v in g.items()}
                if 's' in g and 'D2' in g: rc['phi_sD2'] = g['s'] / (g['s'] + g['D2'])
                r['cls'][cl] = rc
            D['models'][mn] = r
            log(f'{ds} {mn} acc {r["acc"]:.3f} ' + ' | '.join(f'{cl}: ' + ' '.join(f'{Xn} crot {r["cls"][cl].get("Rgroup_crot_dm", {}).get(Xn, float("nan")):.2f} '
                                                                                     f'crw {r["cls"][cl].get("Rgroup_crw", {}).get(Xn, float("nan")):.2f} '
                                                                                     f'crwsm {r["cls"][cl].get("Rgroup_crw_sm", {}).get(Xn, float("nan")):.2f} '
                                                                                     f'crwms {r["cls"][cl].get("Rgroup_crw_ms", {}).get(Xn, float("nan")):.2f}'
                                                                                     for Xn in ('s', 'D2', 'tau21')) for cl in cls_eval))
        S['datasets'][ds] = D
    if tests:
        hp = H.holm([p for _, p in tests])
        S['holm']['shape_edit_vs_crw'] = dict(n_tests=len(tests), n_below_0p05=int(np.sum(np.array(hp) < 0.05)),
                                              p={k: float(v) for (k, _), v in zip(tests, hp)})
        log(f'Holm edit vs crw: {S["holm"]["shape_edit_vs_crw"]["n_below_0p05"]} of {len(tests)} below 0.05')
    (OUT / 'h69_verify.json').write_text(json.dumps(S, indent=1, default=float)); log('wrote h69_verify.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', required=True, choices=['edits', 'eval', 'summary'])
    ap.add_argument('--ds', default=None, choices=['jetclass', 'top'])
    a = ap.parse_args(); log(f'H69v part {a.part} ds {a.ds}'); WORK.mkdir(parents=True, exist_ok=True)
    if a.part == 'edits': part_edits(a.ds)
    elif a.part == 'eval': X.part_eval(a.ds)
    else: part_summary()
    log('done')
