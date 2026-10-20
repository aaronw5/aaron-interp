"""H36 verification, second pass (loop D): axis vs direction-averaged random shift on a third fresh sample, against an ensemble of
weight-permuted networks (complements interp/h36_verify.py = job d-h36-verify, seeds 3611-3613, R1/R2/ANTI/ROT controls).

Fresh samples (new seeds; rows that were in the H36 main sample are dropped):
  top  h4_top_wmass._sample(2600, 3621);  py  h14 load_sample('py', 2600, 3622);  hw  load_sample('hw', 2600, 3623).
Conditions per jet (inputs as h36_axis_top_qg: the axis enters only through (deta, dphi) and dR):
  E (unedited), WTA, HP (inputs recomputed about the axis), R (one uniformly random direction, same per-jet size, as H36),
  Rbar (6 directions at the jet's own offset size, equally spaced with a random per-jet phase: per-jet direction-averaged flip
  probability; lower-variance version of R), M (mirror: the same offset with the opposite sign, i.e. the axis moved away from the
  hardest particle by the same amount).
Nulls: an ensemble of weight-permuted networks: Top ParT seeds 0-5; QG ParT_kin seeds 0-5, PN_kin seeds 0-2, PFN seeds 0-2
  (a permuted network whose calls change in < 0.5 % of jets under Rbar is counted as degenerate and left out of the null band).
LorentzNet and LR(N) are not rescored: their input paths take no axis argument (0 change by construction in H36).
Statistics: % calls changed with 95 % bootstrap CI (B = 2000); axis - Rbar per-jet paired difference (bootstrap CI, normal p);
  McNemar axis vs R and axis vs M; trained - permuted difference of (axis - Rbar) on the same jets (bootstrap CI, normal p);
  Holm across cells per family. Accuracy / AUC before -> after.
Run: ./part_job.sh d-h36-verify2 'python -m interp.h36_verify2'      (H36V_SMOKE=1: 100 per class)
Outputs: $AI_RESULTS/physics/h36_verify2.json, h36_verify2_perjet.npz
"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import h36_axis_top_qg as H  # noqa: E402  (patches h5_qg_counting.kinematics with the axis/shift override)

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
SMOKE = bool(os.environ.get('H36V_SMOKE'))
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
TAG = 'h36_verify2_smoke' if SMOKE else 'h36_verify2'
PER = 100 if SMOKE else 2600
NDIRB = 6
B = 300 if SMOKE else 2000
H.PER_MAIN, H.PER_CAL = PER, 0
H.SEEDS = dict(top=3621, py=3622, hw=3623)
H.NTRAIN_LR = 2000 if SMOKE else 10000
AX = ('WTA', 'HP')


def perm_(tensors, seed):
    import torch
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in tensors:
            f = p.data.view(-1); f.copy_(f[torch.randperm(f.numel(), generator=g)])


def perm_flow(fl, seed):
    import copy
    f2 = copy.deepcopy(fl); rng = np.random.default_rng(seed)
    pm = lambda a: rng.permutation(a.ravel()).reshape(a.shape)
    f2.phi = [(pm(K), pm(b)) for K, b in f2.phi]; f2.F = [(pm(K), pm(b)) for K, b in f2.F]; f2.out = (pm(f2.out[0]), pm(f2.out[1]))
    return f2


def norm_p(d):
    from scipy.stats import norm
    se = d.std(ddof=1) / np.sqrt(len(d))
    return float(2 * norm.sf(abs(d.mean()) / se)) if se > 0 else (1.0 if d.mean() == 0 else 0.0)


def run(ds, S, scorer, names, sign_top, perm_of):
    old = np.load(RES / 'h36_perjet.npz')[f'{ds}|idx']
    keep = np.flatnonzero(~np.isin(S['idx'], old)); y = S['y'][keep]
    log(f'[{ds}] {len(S["y"])} jets, {len(S["y"]) - len(keep)} in the H36 main sample dropped -> {len(keep)} ({np.bincount(y)})')
    off = {a: (S['off'][a][0][keep], S['off'][a][1][keep]) for a in AX}; rr = {a: np.hypot(*off[a]) for a in AX}
    sg = np.where(S['jeta'][keep] < 0, -1.0, 1.0); rng = np.random.default_rng(36900 + ['top', 'py', 'hw'].index(ds))
    sc = {'E': scorer(keep, None)}
    for a in AX:
        sc[a] = scorer(keep, ('ax', S['A'][a][0][keep], S['A'][a][1][keep]))
        sc[f'M_{a}'] = scorer(keep, ('sh', sg * off[a][0], off[a][1]))
        th = rng.uniform(0, 2 * np.pi, len(keep)); sc[f'R_{a}'] = scorer(keep, ('sh', rr[a] * np.cos(th), rr[a] * np.sin(th)))
        ph = rng.uniform(0, 2 * np.pi, len(keep))
        for k in range(NDIRB):
            t = ph + 2 * np.pi * k / NDIRB; sc[f'Rb{k}_{a}'] = scorer(keep, ('sh', rr[a] * np.cos(t), rr[a] * np.sin(t)))
        log(f'[{ds}] scored {a}')
    J = len(keep); bi = np.random.default_rng(36950).integers(0, J, (B, J))
    ci = lambda x: (np.percentile(x[bi].mean(1), [2.5, 97.5]) * 100).tolist()
    out = dict(n=int(J), n_by_class=np.bincount(y).tolist(), dropped_overlap=int(len(S['y']) - J), axinfo=S['axinfo'],
               offsets={a: dict(median_mrad=float(1e3 * np.median(rr[a])),
                                by_class={str(c): float(1e3 * np.median(rr[a][y == c])) for c in (0, 1)}) for a in AX}, models={})
    D = {}
    for nm in names:
        l0 = sc['E'][nm]; c0 = l0 > 0; ytgt = (y == 1) if sign_top else (y == 0); so = (lambda l: l) if sign_top else (lambda l: -l)
        Mm = dict(acc=float(np.mean(c0 == ytgt)), auc=H.auc(so(l0), (y == 1).astype(int)), axes={})
        for a in AX:
            ch = (sc[a][nm] > 0) != c0; chR = (sc[f'R_{a}'][nm] > 0) != c0; chM = (sc[f'M_{a}'][nm] > 0) != c0
            pb = np.mean([(sc[f'Rb{k}_{a}'][nm] > 0) != c0 for k in range(NDIRB)], 0)
            d = ch.astype(float) - pb; D[(nm, a)] = d
            Mm['axes'][a] = dict(pct=float(100 * ch.mean()), ci=ci(ch.astype(float)), pct_R=float(100 * chR.mean()), ci_R=ci(chR.astype(float)),
                                 pct_Rbar=float(100 * pb.mean()), ci_Rbar=ci(pb), pct_M=float(100 * chM.mean()), ci_M=ci(chM.astype(float)),
                                 axis_minus_Rbar_pp=float(100 * d.mean()), axis_minus_Rbar_ci=ci(d), p_axis_vs_Rbar=norm_p(d),
                                 axis_minus_R_pp=float(100 * (ch.mean() - chR.mean())), mcnemar_axis_vs_R=H.H4._mcnemar(chR, ch),
                                 axis_minus_M_pp=float(100 * (ch.mean() - chM.mean())), mcnemar_axis_vs_M=H.H4._mcnemar(chM, ch),
                                 by_class={str(c): dict(axis=float(100 * ch[y == c].mean()), Rbar=float(100 * pb[y == c].mean()),
                                                        M=float(100 * chM[y == c].mean())) for c in (0, 1)},
                                 acc_after=float(np.mean((sc[a][nm] > 0) == ytgt)), auc_after=H.auc(so(sc[a][nm]), (y == 1).astype(int)),
                                 mean_dl=float((sc[a][nm] - l0).mean()), mean_dl_M=float((sc[f'M_{a}'][nm] - l0).mean()),
                                 mean_abs_dl=float(np.abs(sc[a][nm] - l0).mean()),
                                 mean_abs_dl_Rbar=float(np.mean([np.abs(sc[f'Rb{k}_{a}'][nm] - l0).mean() for k in range(NDIRB)])))
        out['models'][nm] = Mm
        log(f'[{ds}] {nm:16s} acc {Mm["acc"]:.4f} AUC {Mm["auc"]:.4f} | ' + ' | '.join(
            f'{a}: {v["pct"]:.2f} R {v["pct_R"]:.2f} Rbar {v["pct_Rbar"]:.2f} M {v["pct_M"]:.2f} ax-Rbar {v["axis_minus_Rbar_pp"]:+.2f} '
            f'{np.round(v["axis_minus_Rbar_ci"], 2)} p {v["p_axis_vs_Rbar"]:.1e}' for a, v in Mm['axes'].items()))
    for nm, pn in perm_of.items():
        for a in AX:
            dd = D[(nm, a)] - D[(pn, a)]
            out['models'][nm]['axes'][a]['vs_perm'] = dict(perm=pn, diff_pp=float(100 * dd.mean()), ci=ci(dd), p=norm_p(dd))
            log(f'[{ds}] {nm} - {pn} {a}: (axis - Rbar) diff {100 * dd.mean():+.2f} {np.round(ci(dd), 2)} p {norm_p(dd):.1e}')
    band = {a: [out['models'][nm]['axes'][a]['axis_minus_Rbar_pp'] for nm in names if 'perm' in nm and out['models'][nm]['axes'][a]['pct_Rbar'] >= 0.5]
            for a in AX}
    out['null_band'] = {a: dict(values=band[a], min=float(min(band[a])) if band[a] else None, max=float(max(band[a])) if band[a] else None,
                                n=len(band[a]), sd=float(np.std(band[a], ddof=1)) if len(band[a]) > 1 else None) for a in AX}
    log(f'[{ds}] null band (axis - Rbar, non-degenerate permuted nets): ' + str({a: np.round(band[a], 2).tolist() for a in AX}))
    pj = {f'{ds}|y': y, f'{ds}|idx': S['idx'][keep]}
    for a in AX: pj[f'{ds}|r_{a}'] = rr[a].astype(np.float32)
    for k, v in sc.items():
        for nm, x in v.items(): pj[f'{ds}|{k}|{nm}'] = np.asarray(x, np.float32)
    return out, pj


def main():
    import torch
    RES.mkdir(parents=True, exist_ok=True); R = dict(node='H36 verify', smoke=SMOKE, per_class=PER, ndir_rbar=NDIRB, datasets={}); PJ = {}
    # ---------------- Top
    S = H.load_top(); log('top loaded', S['axinfo'])
    TM = H.TopModels()
    for sd in range(1, 6):
        p1 = H.H4.WeaverNet('ParT', RES / 'h4_ckpt_part_kin.pt', 'cpu'); perm_(list(p1.model.parameters()), sd); p1.model.to(TM.dev); p1.dev = TM.dev
        TM.w[f'ParT_top_perm{sd}'] = p1
    tnames = ['ParT_top', 'PN_top', 'PFN_top', 'EFN_top', 'ParT_top_perm'] + [f'ParT_top_perm{sd}' for sd in range(1, 6)]

    def tsc(rows, cond):
        P4, nn = S['P4'][rows], S['nn'][rows]; out = {k: np.zeros(len(rows)) for k in tnames}
        for s in range(0, len(rows), 2000):
            sl = slice(s, s + 2000); c = None if cond is None else (cond[0], cond[1][sl], cond[2][sl])
            Jx = H.top_J(P4[sl], nn[sl], c)
            for b in range(0, len(nn[sl]), 512):
                Jb = {k: (v[b:b + 512] if np.ndim(v) else v) for k, v in Jx.items()}; pts, x, v, m = H.H4.weaver_inputs(Jb)
                dv = lambda q: torch.from_numpy(q).to(TM.dev)
                with torch.no_grad():
                    for name, wn in TM.w.items():
                        o = (wn.model(dv(x), v=dv(v), mask=dv(m)) if wn.kind == 'ParT' else wn.model(dv(pts), dv(x), dv(m))).float().cpu().numpy()
                        out[name][s + b:s + b + len(o)] = o[:, 0] - o[:, 1]
            k3 = np.stack([Jx['pt'][:, :128], Jx['deta'][:, :128], Jx['dphi'][:, :128]], -1).astype(np.float16).astype(np.float32)
            for name, f in TM.flows.items(): out[name][sl] = f.score(k3)
        return out
    R['datasets']['top'], pj = run('top', S, tsc, tnames, True,
                                   {'ParT_top': 'ParT_top_perm', 'PN_top': 'ParT_top_perm', 'PFN_top': 'ParT_top_perm', 'EFN_top': 'ParT_top_perm'})
    PJ.update(pj); del TM, S
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    # ---------------- QG
    QS = {g: H.load_qg(g) for g in ('py', 'hw')}
    M = H.load_qg_models()
    extra = [('part_kin', f'part_kin_perm{sd}', sd) for sd in range(1, 6)] + [(s, f'{s}_perm{sd}', sd) for s in ('pnet_kin', 'pfn_k') for sd in range(3)]
    for src, nm, seed in extra:
        import copy
        kind, arch, pidfeat, net = M.m[src]; n2 = copy.deepcopy(net).cpu(); perm_(list(n2.parameters()), seed)
        M.m[nm] = (kind, arch, pidfeat, n2.eval().to(M.dev))
    qn = ['part_kin', 'pnet_kin', 'pfn_k', 'efn_k', 'hw_part_kin', 'hw_pnet_kin', 'part_kin_perm'] + [nm for _, nm, _ in extra]
    for g in ('py', 'hw'):
        S = QS[g]; REF = H.QGRefs(g)

        def qsc(rows, cond, S=S, REF=REF):
            Jt = dict(H.sub_jt(S['Jt'], rows)); Jt['_cond'] = cond
            o = {nm: M.logit(nm, Jt) for nm in qn}; o['LR_girth'] = REF.score(Jt)['LR_girth']; return o
        R['datasets'][g], pj = run(g, S, qsc, qn + ['LR_girth'], False,
                                   {'part_kin': 'part_kin_perm', 'pnet_kin': 'pnet_kin_perm0', 'pfn_k': 'pfn_k_perm0', 'efn_k': 'pfn_k_perm0',
                                    'hw_part_kin': 'part_kin_perm', 'hw_pnet_kin': 'pnet_kin_perm0'})
        PJ.update(pj)
    # ---------------- Holm
    cells = [(ds, nm, a) for ds in R['datasets'] for nm in R['datasets'][ds]['models'] for a in AX]
    for key, get in (('holm_axis_vs_Rbar', lambda v: v['p_axis_vs_Rbar']), ('holm_axis_vs_R', lambda v: v['mcnemar_axis_vs_R']['p']),
                     ('holm_axis_vs_M', lambda v: v['mcnemar_axis_vs_M']['p'])):
        for (ds, nm, a), pa in zip(cells, H.H4._holm([get(R['datasets'][ds]['models'][nm]['axes'][a]) for ds, nm, a in cells])):
            R['datasets'][ds]['models'][nm]['axes'][a][key] = float(pa)
    pc = [(ds, nm, a) for ds, nm, a in cells if 'vs_perm' in R['datasets'][ds]['models'][nm]['axes'][a]]
    for (ds, nm, a), pa in zip(pc, H.H4._holm([R['datasets'][ds]['models'][nm]['axes'][a]['vs_perm']['p'] for ds, nm, a in pc])):
        R['datasets'][ds]['models'][nm]['axes'][a]['vs_perm']['holm'] = float(pa)
    R['holm'] = f'{len(cells)} cells for axis vs Rbar / R / M; {len(pc)} cells for trained - permuted'
    (RES / f'{TAG}.json').write_text(json.dumps(R, indent=1, default=float)); np.savez_compressed(RES / f'{TAG}_perjet.npz', **PJ)
    log(f'wrote {TAG}.json')


if __name__ == '__main__':
    main()
