"""H67 verification (loop D): rerun interp.h67_mass_vs_width on fresh samples (seeds 6761 / 6762 / 6763) with two additions.

1 crot_dm: displacement-matched prong rotation. Every prong of a jet is rotated about its own pT-weighted centroid by one angle
  theta with 2 sin(theta / 2) * rho_bar = D(Ew-10) (rho_bar = pT-weighted mean distance of the particles to their prong centroid,
  D = pT-weighted displacement of Ew-10 on the same jet), random sign; then (f, u) are re-solved for Delta m = Delta w = 0 exactly as
  for crot. The original crot uses U(0, 2 pi) angles and moves particles about twice as far as Ew-10.
2 extra reference classifiers (same MLP(32, 32), same disjoint reference jets): prong-sharing features s_K = 1 - sum_k (P_k / P)^2
  from exclusive-kT K = 2 and K = 3 subjets computed on every jet (no label used; s_2 = 2 z (1 - z)), D2 (beta 1), tau21 (beta 1):
  MLP(m, s, pT), MLP(s, pT), MLP(m, D2, pT), MLP(m, tau21, pT), MLP(D2, pT).
Parts: --part edits --ds {hls,jetclass,top} | --part eval --ds {hls_jedi,jetclass,top} | --part summary | --part check
Outputs: $AI_RESULTS/physics/h67_verify/ (work files, h67_summary.json, figures) and $AI_RESULTS/physics/h67_verify.json (check).
"""
import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import h67_mass_vs_width as M  # noqa: E402
from interp import h12_boost_panel as H  # noqa: E402

VOUT = M.RES / 'physics' / 'h67_verify'
M.OUT = VOUT; M.WORK = VOUT / 'work'
M.SEEDS = dict(hls=6761, jetclass=6762, top=6763)
M.EDITS = M.EDITS + ['crot_dm']
M.REFS = {**M.REFS, 'MLP(m,s,pT)': ('m', 's2', 's3', 'jpt'), 'MLP(s,pT)': ('s2', 's3', 'jpt'), 'MLP(m,D2,pT)': ('m', 'D2', 'jpt'),
          'MLP(m,tau21,pT)': ('m', 'tau21', 'jpt'), 'MLP(D2,pT)': ('D2', 'jpt')}
_make, _rf = M.make_edits, M.ref_feats


def make_edits_v(T0, y, ntop, seed):
    out, G = _make(T0, y, ntop, seed)
    rng = np.random.default_rng(seed + 2000); real = G['real']; pt = T0['pt']
    ref_U, ref_inf = out['Ew-10']; D = M.displacement(G, ref_U, pt); D = np.where(ref_inf['conv'], D, np.median(D[ref_inf['conv']]))
    rbar = (pt * G['rho']).sum(1) / np.maximum(pt.sum(1), 1e-300)
    th = 2 * np.arcsin(np.clip(D / np.maximum(2 * rbar, 1e-12), 0, 1)) * rng.choice([-1.0, 1.0], len(y)); tp = th[:, None]
    c = M.gather(G['c'], G['lab']); d = G['x0'] - c
    xr = np.where(real[..., None], c + np.stack([np.cos(tp) * d[..., 0] - np.sin(tp) * d[..., 1], np.sin(tp) * d[..., 0] + np.cos(tp) * d[..., 1]], -1), 0)
    U, inf = M.solve(T0, G, 0.0, 0.0, base_x=xr); out['crot_dm'] = (U, inf)
    M.log(f'crot_dm: converged {inf["conv"].mean():.3f}; displacement median {np.median(M.displacement(G, U, pt)[inf["conv"]]):.4f} '
          f'vs Ew-10 {np.median(D):.4f}; crot {np.median(M.displacement(G, out["crot"][0], pt)[out["crot"][1]["conv"]]):.4f}')
    return out, G


def sharing(T):
    from interp.h41_colour_flow import ktn_batch
    F = M.F4(T); nn = (T['pt'] > 0).sum(1).astype(np.int64); out = {}
    for K in (2, 3):
        lab = ktn_batch(np.ascontiguousarray(F), nn, K); P = np.stack([np.where(lab == k, T['pt'], 0).sum(1) for k in range(K)], 1)
        S = P.sum(1); out[f's{K}'] = np.where(S > 0, 1 - ((P / np.maximum(S, 1e-300)[:, None]) ** 2).sum(1), np.nan)
    return out


def ref_feats_v(T):
    R = _rf(T)
    ok = np.all(np.isfinite(T['pt']) & np.isfinite(T['eta']) & np.isfinite(T['phi']) & np.isfinite(T['m']), 1) & ((T['pt'] > 0).sum(1) >= 3)
    g = int(np.flatnonzero(ok)[0]); T2 = {k: (np.where(ok[:, None], T[k], T[k][g:g + 1]) if k in ('pt', 'eta', 'phi', 'm') else T[k]) for k in T}
    S = M.shapes(T2); S.update(sharing(T2))
    for k, v in S.items(): R[k] = np.where(ok, v, np.nan)
    return R


M.make_edits = make_edits_v; M.ref_feats = ref_feats_v


# ------------------------------------------------------------------------------------------------ check
def boot_ci(x, rng, nb=1000):
    I = rng.integers(0, len(x), (nb, len(x))); return np.percentile(x[I].mean(1), [2.5, 97.5]).tolist()


def check():
    from interp.part_core import CL
    res = dict(constraints={}, sharing={}, crot_dm={}, phi={}, reproduction={}); rng = np.random.default_rng(6799)
    S = json.loads((VOUT / 'h67_summary.json').read_text()); S0 = json.loads((M.RES / 'physics' / 'h67_summary.json').read_text())
    cls_eval = dict(hls=[0, 1, 2, 3, 4], jetclass=[CL.index(c) for c in M.JC_CL], top=[0, 1])
    pvals, pkeys = [], []
    for ds in ('hls', 'jetclass', 'top'):
        if not (M.WORK / f'{ds}_feats.pkl').exists(): continue
        with open(M.WORK / f'{ds}_feats.pkl', 'rb') as fh: RF = pickle.load(fh)
        per = RF['perjet']; classes = RF['classes']; d = np.load(M.WORK / f'{ds}_tables.npz'); y = d['y'].astype(int)
        conv = {e: per[f'{e}__conv'] for e in M.EDITS}
        # 1 independent constraint check (H.jetp4, float32 tables as the networks saw them)
        tab = lambda nm: dict(pt=d[f'{nm}__pt'].astype(np.float64), eta=d[f'{nm}__eta'], phi=d[f'{nm}__phi'], m=d[f'{nm}__m'].astype(np.float64))
        U = tab('U'); p0 = H.jetp4(U); m0 = np.sqrt(np.maximum(p0[:, 3] ** 2 - (p0[:, :3] ** 2).sum(1), 0)); pt0 = np.hypot(p0[:, 0], p0[:, 1])
        ee = np.arcsinh(p0[:, 2] / pt0); ep = np.arctan2(p0[:, 1], p0[:, 0]); N0 = (U['pt'] > 0).sum(1)
        wid = lambda T: (T['pt'] * np.where(T['pt'] > 0, np.hypot(T['eta'] - ee[:, None], H.wrap(T['phi'] - ep[:, None])), 0)).sum(1) / T['pt'].sum(1)
        w0 = wid(U); cc = {}
        tgt = {'Ew+10': (0, .1), 'Ew-10': (0, -.1), 'Ew+20': (0, .2), 'Ew-20': (0, -.2), 'Em+10': (.1, 0), 'Em-10': (-.1, 0), 'crot': (0, 0), 'crot_dm': (0, 0), 'crand_match': (0, -.1)}
        for e, (am, aw) in tgt.items():
            T = tab(e); c = conv[e]; p = H.jetp4(T); m = np.sqrt(np.maximum(p[:, 3] ** 2 - (p[:, :3] ** 2).sum(1), 0)); pt = np.hypot(p[:, 0], p[:, 1])
            pad = T['pt'] <= 0
            cc[e] = dict(conv=float(c.mean()), max_abs_dm_minus_target=float(np.abs(m[c] / m0[c] - 1 - am).max()), max_abs_dpt=float(np.abs(pt[c] / pt0[c] - 1).max()),
                         max_abs_dw_minus_target=float(np.abs(wid(T)[c] / w0[c] - 1 - aw).max()),
                         max_axis_shift=float(max(np.abs(np.arcsinh(p[c, 2] / pt[c]) - ee[c]).max(), np.abs(H.wrap(np.arctan2(p[c, 1], p[c, 0]) - ep[c])).max())),
                         N_changed=int(((T['pt'] > 0).sum(1) != N0).sum()), padded_nonzero=int((np.abs(T['eta'][pad]) + np.abs(T['phi'][pad]) + np.abs(T['m'][pad]) > 0).sum()))
        res['constraints'][ds] = cc
        # 2 prong sharing and opening per edit: Delta log s_K (K = 3 for the top class, 2 otherwise), f and u medians
        E = json.loads((VOUT / f'h67_edits_{ds}.json').read_text())['edits']; top_k = classes.index('t') if 't' in classes else (classes.index('Tbqq') if 'Tbqq' in classes else 1)
        sh = {}
        for e in ('Ew-10', 'Em+10', 'Ew+10', 'Em-10', 'crot', 'crot_dm'):
            r = dict(f_median=E[e]['f_median'], u_median=E[e]['u_median'], displacement_median=E[e]['displacement_median'], by_class={})
            for k in cls_eval[ds]:
                key = 's3' if k == top_k and ds != 'hls' or (ds == 'hls' and k == 4) else 's2'; sel = conv[e] & (y == k)
                if sel.sum() < 20: continue
                dl = np.log(RF[e][key][sel]) - np.log(RF['U'][key][sel])
                r['by_class'][classes[k]] = dict(feature=key, median_dlog_s=float(np.nanmedian(dl)))
            sh[e] = r
        res['sharing'][ds] = sh
        # 3 displacement-matched rotation control on the trained networks, pair 'uneq' jets
        Lz = np.load(M.WORK / f'{ds}_logits.npz') if (M.WORK / f'{ds}_logits.npz').exists() else np.load(M.WORK / 'hls_jedi_logits.npz')
        models = sorted({k.split('__')[0] for k in Lz.files if 'perm' not in k}); cd = {}
        for mn in models:
            lo = {e: M.logodds_all(Lz[f'{mn}__{e}']) for e in ['U'] + M.EDITS}; cd[mn] = {}
            for k in cls_eval[ds]:
                sel = (y == k) & conv['Ew-10'] & conv['Em+10'] & conv['crot'] & conv['crot_dm']; n = int(sel.sum())
                if n < 20: continue
                A = {e: np.abs(lo[e][sel, k] - lo['U'][sel, k]) for e in ('Ew-10', 'Em+10', 'crot', 'crot_dm')}
                I = rng.integers(0, n, (1000, n)); As, Bs, Rs, Ds_ = (A[e][I].sum(1) for e in ('Ew-10', 'Em+10', 'crot', 'crot_dm'))
                dif = A['Ew-10'] - A['crot_dm']; db = dif[I].mean(1)
                p = float(min(1.0, 2 * min((db <= 0).mean(), (db >= 0).mean()) + 1 / 1000))
                ex = lambda a, b, r: np.maximum(a - r, 0) / np.maximum(np.maximum(a - r, 0) + np.maximum(b - r, 0), 1e-300)
                sA, sB, sR, sD = (A[e].sum() for e in ('Ew-10', 'Em+10', 'crot', 'crot_dm'))
                cd[mn][classes[k]] = dict(n=n, mean_abs=dict(Ew=float(A['Ew-10'].mean()), Em=float(A['Em+10'].mean()), crot=float(A['crot'].mean()), crot_dm=float(A['crot_dm'].mean())),
                                          mean_abs_ci=dict(Ew=boot_ci(A['Ew-10'], rng), crot_dm=boot_ci(A['crot_dm'], rng)),
                                          diff_Ew_minus_crot_dm=float(dif.mean()), diff_ci=np.percentile(db, [2.5, 97.5]).tolist(), p_boot=p,
                                          phi=float(sA / (sA + sB)), phi_excess_crot=float(ex(sA, sB, sR)), phi_excess_crot_dm=float(ex(sA, sB, sD)),
                                          phi_excess_crot_dm_ci=np.nanpercentile(ex(As, Bs, Ds_), [2.5, 97.5]).tolist())
                pkeys.append(f'{ds}|{mn}|{classes[k]}'); pvals.append(p)
        res['crot_dm'][ds] = cd
        # 4 phi per model/class vs every reference (fresh sample), and reproduction of the original phi
        D = S['datasets'][ds]['models']; D0 = S0['datasets'][ds]['models']; refs = {m: r for m, r in D.items() if r['role'] == 'ref'}; ph = {}
        for mn, r in D.items():
            if r['role'] == 'null' and not mn.endswith('perm0'): continue
            ph[mn] = dict(role=r['role'], acc=r['acc'], per_class_acc=r['per_class_acc'], cls={})
            for c, st in r['cls'].items():
                s = st.get('uneq', {})
                if s.get('phi') is None: continue
                lo_, hi_ = s['phi_ci']
                ent = dict(n=s['n'], phi=s['phi'], ci=s['phi_ci'], phi_eq=st.get('eq', {}).get('phi'))
                if r['role'] != 'ref':
                    ent['refs_inside'] = [rn for rn, rr in refs.items() if rr['cls'][c].get('uneq', {}).get('phi') is not None and lo_ <= rr['cls'][c]['uneq']['phi'] <= hi_]
                    ent['nearest_refs'] = sorted(((rn, round(rr['cls'][c]['uneq']['phi'], 3)) for rn, rr in refs.items() if rr['cls'][c].get('uneq', {}).get('phi') is not None),
                                                 key=lambda t: abs(t[1] - s['phi']))[:4]
                    o = D0.get(mn, {}).get('cls', {}).get(c, {}).get('uneq', {})
                    if o.get('phi') is not None:
                        ent['orig_phi'] = o['phi']; ent['orig_ci'] = o['phi_ci']; ent['orig_inside_new_ci'] = bool(lo_ <= o['phi'] <= hi_)
                        ent['ci_overlap'] = bool(max(lo_, o['phi_ci'][0]) <= min(hi_, o['phi_ci'][1]))
                ph[mn]['cls'][c] = ent
        res['phi'][ds] = ph
    hp = H.holm(pvals) if pvals else []
    res['holm_Ew_vs_crot_dm'] = dict(n_tests=len(pvals), n_below_0p05=int(np.sum(np.array(hp) < 0.05)), p=dict(zip(pkeys, [float(v) for v in hp])))
    (M.RES / 'physics' / 'h67_verify.json').write_text(json.dumps(res, indent=1, default=float)); M.log('wrote h67_verify.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', required=True, choices=['edits', 'eval', 'summary', 'check'])
    ap.add_argument('--ds', default=None)
    a = ap.parse_args(); VOUT.mkdir(parents=True, exist_ok=True); M.WORK.mkdir(parents=True, exist_ok=True); M.log(f'H67 verify {a.part} {a.ds}')
    if a.part == 'edits': M.part_edits(a.ds)
    elif a.part == 'eval': M.part_eval(a.ds)
    elif a.part == 'summary': M.part_summary()
    else: check()
    M.log('done')
