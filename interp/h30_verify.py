"""H30 verification (loop D). Fresh sample (seed 3002, 1000 q + 1000 g per generator) and two added controls that separate
the two parts of the H30 main edit m(r,k): (i) the pT transfer from the hardest particle to the selected particle and
(ii) the angular placement of the selected particle.
  m  main edit as H30 (pT_i -> kt_k / dR_r, hardest particle gives the difference, particle placed at dR_r along psi0)
  c  H30 control c1 (pT unchanged, rotation at dR0)
  p  pT-only: the same pT_i -> kt_k / dR_r and the same transfer from the hardest particle, but the particle stays at its own
     dR0 along psi0 about the edited jet's axis (no angular move)
  x  angle-only: pT_i unchanged, no transfer, particle moved to dR_r along psi0
Decomposition per cell: (m - base) = (p - base) + (m - p); (m - p) is the angular move at the new pT.
Also recomputes, from the H30 run json, the Herwig-Pythia map difference relative to the map size (low vs high kt).
  python -m interp.h30_verify [--per_class 1000]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from interp import h30_qg_lund_map as M

V = Path('/j-jepa-vol/aaron-interp')
RES = Path(os.environ.get('AI_RESULTS', V / 'results')) / 'physics'
log = M.log
M.SEED = 3002                       # fresh sample and fresh selection streams (module-global, read at call time)
NAMES = ('part_kin', 'pnet_kin', 'hw_part_kin', 'hw_pnet_kin', 'efn_t', 'pfn_t', 'part_kin_perm')


def make_edits(L0, gen):
    meta, P = M.make_edits(L0, gen)           # m, c (c1), r (c2) as H30, with the fresh seed
    sel, dR0, psi0, pTJ = meta['sel'], meta['dR0'], meta['psi0'], meta['pTJ']
    J = len(L0); Q = {}
    for (r, k) in M.CELLS:
        Pp = np.zeros((J, 5)); X = np.zeros((J, 5)); okm = P[M.cn('m', r, k)][:, 4] > 0.5
        for j, a in enumerate(L0):
            i = sel[j]
            if i < 0 or not okm[j]: continue
            pnew = M.KC[k] / M.RC[r]; d = pnew - a[i, 0]; ph = a[0, 0] - d
            b = a.copy(); b[i, 0] = pnew; b[0, 0] = ph; b = M._place(b, i, dR0[j], psi0[j])
            Pp[j] = [b[i, 0], b[i, 1], b[i, 2], b[0, 0], 1]
            b = M._place(a.copy(), i, M.RC[r], psi0[j]); X[j] = [b[i, 0], b[i, 1], b[i, 2], b[0, 0], 1]
        Q[M.cn('p', r, k)] = Pp; Q[M.cn('x', r, k)] = X
    for c in list(P):
        if c[0] == 'r': del P[c]                 # c2 not needed here
    P.update(Q)
    return meta, P


def step_eval(a):
    import torch
    from interp import h5_qg_counting as H
    from interp import h14_qg_generator as H14
    from interp.h24_qg_generator_models import basic_obs
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; RES.mkdir(parents=True, exist_ok=True)
    Mo = H.Models(dev)
    for name, arch in (('part_kin', 'part'), ('pnet_kin', 'pn')): Mo.add_weaver(name, arch, False)
    Mo.add_permuted('part_kin_perm', 'part_kin')
    for kind in ('efn', 'pfn'):
        net = H.make_ef(kind); net.load_state_dict(torch.load(V / 'models' / 'qg' / f'h5_{kind}_torch.pt', map_location='cpu'))
        Mo.add_torch(f'{kind}_t', net, json.loads((V / 'models' / 'qg' / f'h5_{kind}_torch.json').read_text()))
    for name, arch in (('part_kin', 'part'), ('pnet_kin', 'pn')):
        H14.add_weaver_dir(Mo, 'hw_' + name, arch, False, V / 'models' / 'qg_herwig' / name)
    chk = {}
    for gen in a.gens.split(','):
        L0, y, info = M.load(gen, a.per_class)
        old = np.load(RES / f'h30_edits_{gen}.npz')['idx']; ov = int(np.isin(info['idx'], old).sum())
        hard0 = all(x[0, 0] >= x[:, 0].max() for x in L0)
        meta, P = make_edits(L0, gen); s = meta['sel'] >= 0
        # conservation checks on the main and pT-only edits
        dsum, dN = 0.0, 0
        for c in P:
            if c[0] not in 'mp': continue
            Le = M.apply(L0, meta['sel'], P[c]); okc = P[c][:, 4] > 0.5
            for j in np.flatnonzero(okc):
                dsum = max(dsum, abs(Le[j][:, 0].sum() - L0[j][:, 0].sum())); dN = max(dN, abs(len(Le[j]) - len(L0[j])))
        chk[gen] = dict(J=len(y), overlap_with_h30_sample=ov, hardest_is_index0=bool(hard0), selected=int(s.sum()),
                        max_abs_dsumpt_GeV=float(dsum), max_abs_dN=int(dN))
        log(gen, chk[gen])
        out = dict(y=y, idx=info['idx'], **meta)
        for c in ['base'] + [c for c in P if P[c][:, 4].any()]:
            Le = L0 if c == 'base' else M.apply(L0, meta['sel'], P[c]); Jt = H.from_list(Le, Lpad=180)
            for nm in NAMES: out[f'{c}|{nm}'] = Mo.logit(nm, Jt).astype(np.float32)
            O = basic_obs(Le); out.update({f'obs|{c}|{k}': v.astype(np.float32) for k, v in O.items() if k in ('girth', 'ptd')})
            if c != 'base': out[f'ok|{c}'] = P[c][:, 4] > 0.5
            if c[0] == 'm' and c[2] in '25': log(gen, c, 'done')
        np.savez_compressed(RES / f'h30_verify_logits_{gen}.npz', **out)
    (RES / 'h30_verify_checks.json').write_text(json.dumps(chk, indent=1))


def bh(ps):
    return M.bh(ps)


def step_stats(a):
    from interp import h5_qg_counting as H
    from interp.h14_qg_generator import Boot
    R = dict(sample=dict(per_class=a.per_class, seed=3002), checks=json.loads((RES / 'h30_verify_checks.json').read_text()), gens={})
    for gen in a.gens.split(','):
        Z = np.load(RES / f'h30_verify_logits_{gen}.npz'); y = Z['y'].astype(int); J = len(y); Bt = Boot(J, B=1000, P=2000, seed=3002)
        g = dict(unedited={nm: H.roc_metrics(Z[f'base|{nm}'].astype(float), y) for nm in NAMES}, maps={}, summary={})
        cells = [(r, k) for r, k in M.CELLS if f'ok|m{r}{k}' in Z.files and Z[f'ok|m{r}{k}'].sum() >= min(200, J // 4)]
        g['cells'] = [f'{r}{k}' for r, k in cells]
        for nm in NAMES:
            b0 = Z[f'base|{nm}'].astype(float); mp = {}
            for (r, k) in cells:
                o = Z[f'ok|m{r}{k}'] & Z[f'ok|c{r}{k}'] & Z[f'ok|p{r}{k}'] & Z[f'ok|x{r}{k}']
                v = {t: Z[f'{t}{r}{k}|{nm}'].astype(float) - b0 for t in 'mcpx'}
                terms = dict(main_c1=v['m'] - v['c'], pt_only=v['p'], angle_at_new_pt=v['m'] - v['p'], angle_only=v['x'], c1=v['c'])
                e = dict(n=int(o.sum()), dR=float(M.RC[r]), kt=float(M.KC[k]))
                for tn, d in terms.items():
                    ci, _ = Bt.mean_ci(d, o)
                    e[tn] = dict(mean=float(d[o].mean()), ci=ci, p=Bt.signflip(d, o),
                                 q=float(d[o & (y == 1)].mean()), g=float(d[o & (y == 0)].mean()))
                for ob in ('ptd', 'girth'):
                    e[f'd{ob}_pt_only'] = float((Z[f'obs|p{r}{k}|{ob}'] - Z[f'obs|base|{ob}'])[o].mean())
                    e[f'd{ob}_main_c1'] = float((Z[f'obs|m{r}{k}|{ob}'] - Z[f'obs|c{r}{k}|{ob}'])[o].mean())
                mp[f'{r}{k}'] = e
            for tn in ('main_c1', 'pt_only', 'angle_at_new_pt', 'angle_only'):
                for key, q in zip(mp, bh([mp[c][tn]['p'] for c in mp])): mp[key][tn]['bh'] = float(q)
            g['maps'][nm] = mp
            Y = {tn: np.array([mp[c][tn]['mean'] for c in mp]) for tn in ('main_c1', 'pt_only', 'angle_at_new_pt', 'angle_only')}
            dptd = np.array([mp[c]['dptd_main_c1'] for c in mp])
            sm = dict(corr_mainc1_vs_ptonly=float(np.corrcoef(Y['main_c1'], Y['pt_only'])[0, 1]),
                      corr_mainc1_vs_angle_at_new_pt=float(np.corrcoef(Y['main_c1'], Y['angle_at_new_pt'])[0, 1]),
                      var_share_ptonly=float(1 - ((Y['main_c1'] - Y['pt_only']) ** 2).sum() / ((Y['main_c1'] - Y['main_c1'].mean()) ** 2).sum()),
                      R2_mainc1_on_dptd=float(np.corrcoef(Y['main_c1'], dptd)[0, 1] ** 2))
            # dR slope at kt 1.29 GeV (k = 2): cell dR 0.026 minus cell dR 0.312, per term, bootstrap CI over jets
            if '02' in mp and '52' in mp:
                o = (Z['ok|m02'] & Z['ok|c02'] & Z['ok|p02'] & Z['ok|x02'] & Z['ok|m52'] & Z['ok|c52'] & Z['ok|p52'] & Z['ok|x52'])
                for tn, f in (('main_c1', lambda t: (Z[f'm{t}|{nm}'] - Z[f'c{t}|{nm}'])), ('pt_only', lambda t: (Z[f'p{t}|{nm}'] - b0)),
                              ('angle_at_new_pt', lambda t: (Z[f'm{t}|{nm}'] - Z[f'p{t}|{nm}']))):
                    d = (f('02') - f('52')).astype(float); ci, _ = Bt.mean_ci(d, o)
                    sm[f'slope_kt1.29_dR0.026_minus_0.312|{tn}'] = dict(mean=float(d[o].mean()), ci=ci, p=Bt.signflip(d, o), n=int(o.sum()))
            g['summary'][nm] = sm
            log(gen, nm, json.dumps(sm)[:600])
        # Herwig-trained minus Pythia-trained: absolute and relative (|diff| / mean |map|) low vs high kt
        for mh, mpy in (('hw_part_kin', 'part_kin'), ('hw_pnet_kin', 'pnet_kin')):
            low = np.array([M.KC[k] < 1.0 for r, k in cells])
            A = np.array([g['maps'][mh][f'{r}{k}']['main_c1']['mean'] for r, k in cells])
            Bm = np.array([g['maps'][mpy][f'{r}{k}']['main_c1']['mean'] for r, k in cells])
            dif = np.abs(A - Bm); sz = 0.5 * (np.abs(A) + np.abs(Bm)); rel = dif / np.maximum(sz, 1e-6)
            g['summary'][f'{mh} - {mpy}'] = dict(abs_low=float(dif[low].mean()), abs_high=float(dif[~low].mean()),
                                                 rel_low=float(np.median(rel[low])), rel_high=float(np.median(rel[~low])),
                                                 size_low=float(sz[low].mean()), size_high=float(sz[~low].mean()))
            log(gen, mh, '-', mpy, g['summary'][f'{mh} - {mpy}'])
        R['gens'][gen] = g
    # same relative contrast on the original H30 run
    try:
        O = json.loads((RES / 'h30_run.json').read_text()); orig = {}
        for gen, G in O['gens'].items():
            for mh, mpy in (('hw_part_kin', 'part_kin'), ('hw_pnet_kin', 'pnet_kin')):
                ks = [f'{r}{k}' for r, k in G['cells']]; low = np.array([M.KC[int(c[1])] < 1.0 for c in ks])
                A = np.array([G['maps'][mh][c]['diff'] for c in ks]); Bm = np.array([G['maps'][mpy][c]['diff'] for c in ks])
                dif = np.abs(A - Bm); sz = 0.5 * (np.abs(A) + np.abs(Bm)); rel = dif / np.maximum(sz, 1e-6)
                orig[f'{gen}|{mh} - {mpy}'] = dict(abs_low=float(dif[low].mean()), abs_high=float(dif[~low].mean()),
                                                   rel_low=float(np.median(rel[low])), rel_high=float(np.median(rel[~low])),
                                                   size_low=float(sz[low].mean()), size_high=float(sz[~low].mean()))
        R['h30_run_relative_contrast'] = orig; log('orig relative', orig)
    except Exception as ex:
        log('orig relative contrast failed', ex)
    (RES / 'h30_verify.json').write_text(json.dumps(R, indent=1, default=float)); log('wrote', RES / 'h30_verify.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', default='all', choices=['eval', 'stats', 'all'])
    ap.add_argument('--gens', default='py,hw'); ap.add_argument('--per_class', type=int, default=1000)
    a = ap.parse_args()
    if a.step in ('eval', 'all'): step_eval(a)
    if a.step in ('stats', 'all'): step_stats(a)
