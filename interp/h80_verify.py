"""H80 verification (loop D): fresh-sample rerun of the NS6 vs FREE comparison, plus a profile-matched control (FREEM).

Why FREEM. The FREE arm moves every particle by an i.i.d. Gaussian amount in normalised units. The NS6 arm projects the same
vector onto the null space of 19 constraints (15 basis values + m, pT, y_J, phi_J) and Gauss-Newton corrects it, so its
per-particle displacement profile differs (H80: |dlog pT^D| is 1.4-1.9x larger under NS6 than under FREE). A ratio
kappa = |ds(NS6)| / |ds(FREE)| therefore mixes "the basis is held" with "a different set of particles is moved".
FREEM takes the realised NS6 displacement of each jet, |du_NS6| per particle and coordinate, gives every entry a random
sign (tied particles keep a common angular move), projects it onto the FREE constraints (m, pT, axis) and rescales it to the
same total norm as du_NS6, then solves back onto m, pT, axis. It moves the same particles by the same amounts in the same
coordinates as NS6, but the basis is free to change.
  kappa_m = mean|ds(NS6)| / mean|ds(FREEM)|   (basis held vs not held, displacement profile matched)
Fresh samples: seeds top 9101, jc 9102, qg 9103 (H80 used 8001-8003); eps 0.01 and 0.00125.
Run (GPU): ./part_job.sh d-h80-verify 'python -m interp.h80_verify --part run --ds top,qg,jc && python -m interp.h80_verify --part stats'
Outputs: $AI_RESULTS/physics/h80_verify.json (stats); per-dataset work files in $AI_RESULTS/physics/h80_verify_work/.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from interp import h80_nsub_basis_nullspace as H

RES0 = Path(os.environ.get('AI_RESULTS', H.V / 'results')) / 'physics'
WORK = RES0 / 'h80_verify_work'
H.RES = WORK
H.SEEDS = dict(top=9101, jc=9102, qg=9103, hls=9104)
H.EPS = (0.01, 0.00125)
H.ARMS = dict(NS6=6, FREE=1, FREEM=1)
EXTRA = {}


def make_edits_v(T, O0, seed, dev, chunk=1000):
    import torch
    Sv = H.Solver(dev); J = len(T['pt']); rows_all = np.arange(J); E = {}; info = {}
    conds = [f'{a}@{e}' for a in ('NS6', 'FREE', 'FREEM') for e in H.EPS] + [f'FLOOR@{H.FLOOR_EPS}']
    for c in conds:
        E[c] = dict(pt=T['pt'].copy(), eta=T['eta'].copy(), phi=T['phi'].copy(), m=T['m'], ax_eta=T['ax_eta'], ax_phi=T['ax_phi'], ext=T['ext'])
        info[c] = dict(res=np.zeros(J), rms_dlogpt=np.zeros(J), rms_dr=np.zeros(J), n_recluster=np.zeros(J), n_tied=np.zeros(J))
        EXTRA[c] = dict(hard_share=np.full(J, np.nan), top4_share=np.full(J, np.nan), norm=np.full(J, np.nan))
    for s in range(0, J, chunk):
        rows = rows_all[s:s + chunk]; S = Sv.pack(T, rows, O0['lab']); n = S['n']; u0 = S['u']; A = S['A']; mk3 = S['mk3']
        rng = np.random.default_rng([seed, 80, s])
        v = torch.from_numpy(rng.standard_normal(u0.shape)).to(dev) * mk3; nn = S['mk'].sum(1).double()
        tie = lambda w: torch.cat([w[:, :n], (A @ w[:, n:2 * n, None])[..., 0], (A @ w[:, 2 * n:, None])[..., 0]], 1) * mk3
        v = tie(v); lab0 = S['lab'].copy()
        mk = S['mk'].cpu().numpy(); pt0 = np.exp(u0[:, :n].cpu().numpy() * H.SA) * mk; z = pt0 / pt0.sum(1, keepdims=True)
        o4 = np.argsort(-pt0, 1)[:, :4]; top4 = np.zeros_like(mk, dtype=bool); np.put_along_axis(top4, o4, True, 1)

        def store(c, u, r, nre):
            info[c]['n_recluster'][rows] = nre
            pt, eta, phi = [q.detach().cpu().numpy() for q in Sv.coords(S, u)]
            Ec = E[c]; sl = (rows, slice(0, n))
            Ec['pt'][sl] = np.where(mk, pt, 0); Ec['eta'][sl] = np.where(mk, eta, 0); Ec['phi'][sl] = np.where(mk, (phi + np.pi) % (2 * np.pi) - np.pi, 0)
            du = (u - u0).detach().cpu().numpy(); cnt = np.maximum(mk.sum(1), 1)
            info[c]['res'][rows] = r.cpu().numpy(); info[c]['n_tied'][rows] = S['n_tied'].cpu().numpy()
            info[c]['rms_dlogpt'][rows] = np.sqrt(((du[:, :n] * H.SA) ** 2).sum(1) / cnt)
            info[c]['rms_dr'][rows] = np.sqrt((((du[:, n:2 * n] * H.SANG) ** 2) + (du[:, 2 * n:] * H.SANG) ** 2).sum(1) / cnt)
            q = du[:, :n] ** 2 + du[:, n:2 * n] ** 2 + du[:, 2 * n:] ** 2; tot = np.maximum(q.sum(1), 1e-30)
            EXTRA[c]['hard_share'][rows] = (z * q).sum(1) / tot * cnt     # pT-weighted share x N: 1 = move independent of pT
            EXTRA[c]['top4_share'][rows] = (q * top4).sum(1) / tot; EXTRA[c]['norm'][rows] = np.sqrt(tot)

        def ns_solve(u_start, c0):
            Sv.set_labels(S, lab0); u, r = Sv.solve(S, u_start, c0, 6); nre = np.zeros(len(rows), int)
            for _ in range(3):
                lab1 = Sv.recluster(S, u); ch = np.any(lab1[:, 2:6] != S['lab'][:, 2:6], axis=(1, 2))
                if not ch.any(): break
                nre += ch; Sv.set_labels(S, lab1); u, r = Sv.solve(S, u, c0, 6)
            return u, r, nre

        Sv.set_labels(S, lab0); c6, _ = Sv.jac(S, u0, 6); c1, _ = Sv.jac(S, u0, 1)
        vN = Sv.project(S, u0, v, 6); vN = vN * (torch.sqrt(3 * nn) / vN.norm(dim=1).clamp_min(1e-30))[:, None]
        vF = Sv.project(S, u0, v, 1); vF = vF * (torch.sqrt(3 * nn) / vF.norm(dim=1).clamp_min(1e-30))[:, None]
        for e in H.EPS:
            u, r, nre = ns_solve(u0 + vN * (e / 0.01), c6); store(f'NS6@{e}', u, r, nre); duN = (u - u0).detach()
            Sv.set_labels(S, lab0); uf, rf = Sv.solve(S, u0 + vF * (e / 0.01), c1, 1); store(f'FREE@{e}', uf, rf, np.zeros(len(rows), int))
            rs = np.random.default_rng([seed, 81, s, int(round(e * 1e6))])
            sg = torch.from_numpy(rs.choice(np.array([-1.0, 1.0]), duN.shape)).to(dev)
            w = tie(duN.abs() * sg); w = Sv.project(S, u0, w, 1)
            w = w * (duN.norm(dim=1) / w.norm(dim=1).clamp_min(1e-30))[:, None]
            um, rm = Sv.solve(S, u0 + w, c1, 1); store(f'FREEM@{e}', um, rm, np.zeros(len(rows), int))
        Sv.set_labels(S, lab0); ufl, rfl = Sv.solve(S, u0 + vF * (H.FLOOR_EPS / 0.01), c1, 1)
        store(f'FLOOR@{H.FLOOR_EPS}', ufl, rfl, np.zeros(len(rows), int))
        H.log(f'  verify edits {s + len(rows)}/{J}')
    return E, info


def part_run(a):
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; WORK.mkdir(parents=True, exist_ok=True); H.log('device', dev)
    H.make_edits = make_edits_v
    for ds in a.ds.split(','):
        EXTRA.clear(); H.run_ds(ds, dev, a.smoke)
        tag = '_smoke' if a.smoke else ''
        np.savez_compressed(WORK / f'h80v_extra_{ds}{tag}.npz', **{f'{k}|{c}': v for c, d in EXTRA.items() for k, v in d.items()})


def part_stats(a):
    tag = '_smoke' if a.smoke else ''; out = dict(seeds=H.SEEDS, eps=list(H.EPS), B=H.NB, P=H.NPERM, ds={}); tests = []
    for ds in a.ds.split(','):
        p = WORK / f'h80_{ds}{tag}.npz'
        if not p.exists(): H.log('missing', p); continue
        Z = dict(np.load(p)); R = json.loads((WORK / f'h80_{ds}{tag}.json').read_text()); X = dict(np.load(WORK / f'h80v_extra_{ds}{tag}.npz'))
        y = Z['y'].astype(int); J = len(y); idx = np.random.default_rng(H.SEEDS[ds] + 100).integers(0, J, (H.NB, J))
        models = sorted({k.split('|')[1] for k in Z if k.startswith('lo|') and k.endswith('|base')})
        G = out['ds'][ds] = dict(n=J, sample=R['sample'], c2st=R['c2st'], edits=R['edits'], models_info=R['models'], profile={}, kappa={})
        for c in [k.split('|')[1] for k in X if k.startswith('hard_share|')]:
            if c.startswith('FLOOR'): okc = Z[f'ok|{c}']
            else:
                e_ = c.split('@')[1]; okc = Z[f'ok|NS6@{e_}'] & Z[f'ok|FREE@{e_}'] & Z[f'ok|FREEM@{e_}']   # jets used for kappa
            G['profile'][c] = {k: float(np.nanmedian(X[f'{k}|{c}'][okc])) for k in ('hard_share', 'top4_share', 'norm')}
        sig = ds in ('top', 'qg')
        for mn in models:
            role = R['models'].get(mn, {}).get('role', 'ref')
            if role == 'shuf' or (role == 'ref' and mn not in ('GBDT_B6', 'MLP_B6', 'GBDT_B7', 'MLP_B7', 'LR_pTD')): continue
            s0 = Z[f'lo|{mn}|base'].astype(float)
            if sig:
                sgn = np.where(y == 1, 1.0, -1.0); thr = np.median((s0 * sgn)[y == 1]); call = lambda s: (s * sgn) > thr
            else:
                thr = np.array([np.median(s0[y == k]) if (y == k).any() else 0 for k in range(y.max() + 1)]); call = lambda s: s > thr[y]
            c0 = call(s0)
            for e in H.EPS:
                ok = Z[f'ok|NS6@{e}'] & Z[f'ok|FREE@{e}'] & Z[f'ok|FREEM@{e}']
                if ok.sum() < 20: continue
                L = {a_: Z[f'lo|{mn}|{a_}@{e}'].astype(float) for a_ in ('NS6', 'FREE', 'FREEM')}
                d = {a_: np.abs(L[a_] - s0) for a_ in L}
                bt = {a_: H._boot(np.where(ok, d[a_], 0), idx) for a_ in d}
                fl = np.abs(Z[f'lo|{mn}|FLOOR@{H.FLOOR_EPS}'].astype(float) - s0)[ok].mean()
                rat = lambda a_, b_: (float(d[a_][ok].mean() / max(d[b_][ok].mean(), 1e-12)),
                                      np.percentile(bt[a_] / np.maximum(bt[b_], 1e-12), [2.5, 97.5]).tolist())
                k, kci = rat('NS6', 'FREE'); km, kmci = rat('NS6', 'FREEM'); mf, mfci = rat('FREEM', 'FREE')
                ch = {a_: call(L[a_]) != c0 for a_ in L}
                r_ = dict(role=role, n_ok=int(ok.sum()), abs_ds={a_: float(d[a_][ok].mean()) for a_ in d}, floor=float(fl),
                          kappa=k, kappa_ci=kci, kappa_m=km, kappa_m_ci=kmci, freem_over_free=mf, freem_over_free_ci=mfci,
                          kappa_m_floorcorr=float((d['NS6'][ok].mean() - fl) / max(d['FREEM'][ok].mean() - fl, 1e-12)),
                          call_change={a_: float(ch[a_][ok].mean()) for a_ in ch}, mcnemar_NS6_vs_FREEM=H.mcnemar(ch['NS6'][ok], ch['FREEM'][ok]))
                if role == 'trained' or mn.lower().startswith('efn') or mn in ('GBDT_B6', 'LR_pTD'):
                    r_['p_kappa_m_eq_1'] = H.signflip((d['NS6'] - d['FREEM'])[ok], seed=3)
                    r_['p_kappa_eq_1'] = H.signflip((d['NS6'] - d['FREE'])[ok], seed=4)
                    if role == 'trained': tests.append((ds, mn, e, 'kappa_m=1', r_['p_kappa_m_eq_1']))
                G['kappa'][f'{mn}@{e}'] = r_
        H.log(ds, ' '.join(f'{k} k {v["kappa"]:.2f} km {v["kappa_m"]:.2f}' for k, v in G['kappa'].items() if k.endswith('@0.01')))
    if tests:
        adj = H.holm([t[-1] for t in tests])
        out['holm'] = [dict(ds=t[0], model=t[1], eps=t[2], test=t[3], p=t[4], p_holm=float(q)) for t, q in zip(tests, adj)]
    (RES0 / f'h80_verify{tag}.json').write_text(json.dumps(out, indent=1, default=float)); H.log('wrote', RES0 / f'h80_verify{tag}.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', required=True, choices=['run', 'stats'])
    ap.add_argument('--ds', default='top,qg,jc'); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    dict(run=part_run, stats=part_stats)[a.part](a)
