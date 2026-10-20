"""H36v (loop D, skeptic of H36): axis convention on Top Landscape and quark-gluon taggers, fresh samples and added controls.

Question. H36 reads "no evidence that the trained networks read the axis beyond a translation", using the axis - R excess
(R = random-direction translation of the same per-jet size) and a weight-permuted ParT as the null. Checks here:
  1. reproduce the % calls changed about the WTA / HP axes on fresh samples (Top seed 3611, QG Py 3612, QG Hw 3613);
  2. direction controls of the same per-jet size: R1, R2 (two random directions; R1 - R2 is the null of the contrast itself),
     ANTI (the axis translation reversed: the hardest particle moved away from the origin instead of onto it), ROT (the axis
     translation rotated by 90 degrees);
  3. architecture-matched permuted nulls: ParT (weight-permutation seeds 0 and 1), PN, PFN, EFN, each with every weight
     tensor permuted (Top and QG);
  4. implementation checks: recompute about the E-scheme axis itself (must equal the unedited score); WTA index against an
     independent brute-force C/A + WTA implementation on 300 jets per sample.
Metrics per model x sample x axis: % calls changed (call = sign of log-odds; gluon-positive for QG, top-positive for Top) with
95 % bootstrap over jets (B = 2000); paired contrasts axis - R1, axis - ANTI, axis - ROT, R1 - R2 (pp, bootstrap CI, exact
McNemar p, Holm over all cells per contrast); mean |dl| per condition and the paired difference axis - mean(R1, R2) with CI.
Run: ./sync.sh && ./part_job.sh d-h36-verify 'python -m interp.h36_verify'   (H36V_SMOKE=1: 100 jets per class)
Output: $AI_RESULTS/physics/h36_verify.json
"""
import copy
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import h36_axis_top_qg as H36   # noqa: E402  (patches h5_qg_counting.kinematics for axis / shift conditions)
from interp import h4_top_wmass as H4       # noqa: E402

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
SMOKE = bool(os.environ.get('H36V_SMOKE'))
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
PER = 100 if SMOKE else 3000
B = 200 if SMOKE else 2000
H36.PER_MAIN, H36.PER_CAL = PER, 0
H36.SEEDS = dict(top=3611, py=3612, hw=3613)
H36.NTRAIN_LR = 2000 if SMOKE else 20000
AXES = ('WTA', 'HP')
TOP_EXTRA = ['ParT_top_perm1', 'PN_top_perm', 'PFN_top_perm', 'EFN_top_perm']
QG_EXTRA = ['part_kin_perm1', 'pnet_kin_perm', 'pfn_k_perm', 'efn_k_perm']
H36.TOP_NETS = H36.TOP_NETS + TOP_EXTRA
H36.QG_NETS = H36.QG_NETS + QG_EXTRA


def permute_torch(net, seed):
    import torch
    net2 = copy.deepcopy(net).cpu(); g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in net2.parameters():
            f = p.data.view(-1); f.copy_(f[torch.randperm(f.numel(), generator=g)])
    return net2


def permute_flow(fl, seed):
    rng = np.random.default_rng(seed); f2 = copy.deepcopy(fl)
    pm = lambda a: a.ravel()[rng.permutation(a.size)].reshape(a.shape)
    f2.phi = [(pm(K), pm(b)) for K, b in fl.phi]; f2.F = [(pm(K), pm(b)) for K, b in fl.F]; f2.out = (pm(fl.out[0]), pm(fl.out[1]))
    return f2


def wta_bruteforce(eta, phi, pt):
    """independent C/A + WTA: explicit pseudojet list (eta, phi, pt, leaf index), merge the closest pair, keep the harder direction"""
    pj = [[float(e), float(p), float(t), i] for i, (e, p, t) in enumerate(zip(eta, phi, pt))]
    while len(pj) > 1:
        best = None
        for a in range(len(pj)):
            for b in range(a + 1, len(pj)):
                d = (pj[a][0] - pj[b][0]) ** 2 + H36.wrap(pj[a][1] - pj[b][1]) ** 2
                if best is None or d < best[0]: best = (d, a, b)
        _, a, b = best; h, s = (a, b) if pj[a][2] >= pj[b][2] else (b, a)
        pj[h][2] += pj[s][2]; del pj[s]
    return pj[0][3]


def stats(sc, y, names, sign_top, off, jeta):
    rngb = np.random.default_rng(36151); J = len(y); bi = rngb.integers(0, J, (B, J)).astype(np.int32)
    ci = lambda x: np.percentile(x[bi].mean(1), [2.5, 97.5]).tolist()
    out = {}
    for nm in names:
        l0 = sc['E'][nm]; c0 = l0 > 0; ytgt = (y == 1) if sign_top else (y == 0)
        orient = (lambda l: l) if sign_top else (lambda l: -l)
        M = dict(acc=float(np.mean(c0 == ytgt)), auc=float(H4._auc(orient(l0), (y == 1).astype(int))),
                 repeat_max_abs_dl=float(np.max(np.abs(sc['E2'][nm] - l0))),
                 Eaxis_recompute_max_abs_dl=float(np.max(np.abs(sc['AX_E'][nm] - l0))), axes={})
        for a in AXES:
            ch = {k: (sc[f'{k}_{a}' if k != 'AX' else a][nm] > 0) != c0 for k in ('AX', 'R1', 'R2', 'ANTI', 'ROT')}
            adl = {k: np.abs(sc[f'{k}_{a}' if k != 'AX' else a][nm] - l0) for k in ('AX', 'R1', 'R2', 'ANTI', 'ROT')}
            V = dict(pct={k: float(100 * v.mean()) for k, v in ch.items()}, ci95={k: ci(100 * v.astype(float)) for k, v in ch.items()},
                     mean_abs_dl={k: float(v.mean()) for k, v in adl.items()},
                     pct_by_class={k: {str(c): float(100 * v[y == c].mean()) for c in (0, 1)} for k, v in ch.items()},
                     acc_after=float(np.mean((sc[a][nm] > 0) == ytgt)), auc_after=float(H4._auc(orient(sc[a][nm]), (y == 1).astype(int))),
                     contrasts={})
            for c1, c2 in (('AX', 'R1'), ('AX', 'R2'), ('AX', 'ANTI'), ('AX', 'ROT'), ('R1', 'R2'), ('ANTI', 'R1')):
                d = 100 * (ch[c1].astype(float) - ch[c2].astype(float)); mc = H4._mcnemar(ch[c2], ch[c1])
                V['contrasts'][f'{c1}-{c2}'] = dict(pp=float(d.mean()), ci95=ci(d), mcnemar_p=mc['p'], n_only_first=mc['fail_to_pass'],
                                                   n_only_second=mc['pass_to_fail'])
            Rm = 0.5 * (adl['R1'] + adl['R2']); dd = adl['AX'] - Rm
            V['mean_abs_dl_AX_minus_R'] = dict(value=float(dd.mean()), ci95=ci(dd),
                                               rel=float(adl['AX'].mean() / Rm.mean()) if Rm.mean() > 0 else None)
            dd2 = adl['AX'] - adl['ANTI']
            V['mean_abs_dl_AX_minus_ANTI'] = dict(value=float(dd2.mean()), ci95=ci(dd2))
            Rc = 0.5 * (ch['R1'].mean() + ch['R2'].mean())
            V['ratio_AX_over_Rmean'] = float(ch['AX'].mean() / Rc) if Rc > 0 else None
            M['axes'][a] = V
        out[nm] = M
        log(f'  {nm:16s} acc {M["acc"]:.4f} AUC {M["auc"]:.4f} Echeck {M["Eaxis_recompute_max_abs_dl"]:.1e} | ' + ' | '.join(
            f'{a}: ' + ' '.join(f'{k} {v:.2f}' for k, v in M['axes'][a]['pct'].items()) +
            f' AX-R1 {M["axes"][a]["contrasts"]["AX-R1"]["pp"]:+.2f} AX-ANTI {M["axes"][a]["contrasts"]["AX-ANTI"]["pp"]:+.2f}'
            f' R1-R2 {M["axes"][a]["contrasts"]["R1-R2"]["pp"]:+.2f}' for a in AXES))
    return out


def measure(ds, S, scorer, names, sign_top):
    J = len(S['y']); rows = np.arange(J); sg = np.where(S['jeta'] < 0, -1.0, 1.0)
    rng = np.random.default_rng(36141 + ['top', 'py', 'hw'].index(ds))
    sc = {'E': scorer(rows, None)}; sc['E2'] = scorer(rows, None)
    sc['AX_E'] = scorer(rows, ('ax', S['jeta'], S['jphi'])); log(f'[{ds}] E, E2, AX_E scored')
    offs = {}
    for a in AXES:
        oe, op = S['off'][a]; r = np.hypot(oe, op); offs[a] = r
        tx, ty = -sg * oe, -op                                         # the axis recompute as a translation of the stored arrays
        sc[a] = scorer(rows, ('ax', S['A'][a][0], S['A'][a][1]))
        for k in ('R1', 'R2'):
            th = rng.uniform(0, 2 * np.pi, J); sc[f'{k}_{a}'] = scorer(rows, ('sh', r * np.cos(th), r * np.sin(th)))
        sc[f'ANTI_{a}'] = scorer(rows, ('sh', -tx, -ty))
        sc[f'ROT_{a}'] = scorer(rows, ('sh', -ty, tx))
        log(f'[{ds}] {a}: axis, R1, R2, ANTI, ROT scored')
    res = dict(n=int(J), n_per_class={str(c): int((S['y'] == c).sum()) for c in (0, 1)}, axinfo=S['axinfo'],
               offset_median_mrad={a: float(1e3 * np.median(offs[a])) for a in AXES},
               models=stats(sc, S['y'], names, sign_top, offs, S['jeta']))
    return res


def wta_check(S, eta_l, phi_l, pt_l, n=300):
    idx = np.random.default_rng(36161).choice(len(eta_l), min(n, len(eta_l)), replace=False); bad = 0
    for j in idx:
        e, p, t = eta_l[j], phi_l[j], pt_l[j]
        iw = H36.wta_index((e, p, t)); ib = wta_bruteforce(e, p, t)
        if not (np.isclose(e[iw], e[ib]) and np.isclose(H36.wrap(p[iw] - p[ib]), 0)): bad += 1
    return dict(n_checked=int(len(idx)), n_mismatch=int(bad))


def overlap(ds, idx):
    f = RES / 'h36_perjet.npz'
    if not f.exists(): return None
    old = np.load(f)[f'{ds}|idx']
    return dict(n_fresh=int(len(idx)), n_in_h36_main=int(np.isin(idx, old).sum()))


def main():
    RES.mkdir(parents=True, exist_ok=True); R = dict(node='H36v', smoke=SMOKE, per_class=PER, B=B, datasets={})
    # Top
    S = H36.load_top(); log(f'top: {len(S["y"])} jets; axes {S["axinfo"]}')
    P = S['P4'].astype(np.float64); pt = np.hypot(P[..., 0], P[..., 1]); nn = S['nn']
    eta = np.arcsinh(P[..., 2] / np.maximum(pt, 1e-12)); phi = np.arctan2(P[..., 1], P[..., 0])
    wc = wta_check(S, [eta[j, :nn[j]] for j in range(len(nn))], [phi[j, :nn[j]] for j in range(len(nn))], [pt[j, :nn[j]] for j in range(len(nn))], 100 if SMOKE else 300)
    log('top WTA check', wc)
    TM = H36.TopModels()
    import torch
    pn = H4.WeaverNet('PN', H36.RES / 'h4_ckpt_pnet_kin.pt', 'cpu')
    for name, src, seed in (('ParT_top_perm1', 'ParT', 1), ('PN_top_perm', 'PN', 0)):
        w = H4.WeaverNet(src, H36.RES / ('h4_ckpt_part_kin.pt' if src == 'ParT' else 'h4_ckpt_pnet_kin.pt'), 'cpu')
        w.model = permute_torch(w.model, seed).eval().to(TM.dev); w.dev = TM.dev; TM.w[name] = w
    del pn
    TM.flows['PFN_top_perm'] = permute_flow(TM.flows['PFN_top'], 0); TM.flows['EFN_top_perm'] = permute_flow(TM.flows['EFN_top'], 0)
    ordr = {}

    def top_scorer(rows, cond):
        if len(rows) not in ordr: ordr[len(rows)] = np.random.default_rng(605).permutation(len(rows))
        return TM.score(S['P4'][rows], S['nn'][rows], cond, ordr[len(rows)])
    R['datasets']['top'] = measure('top', S, top_scorer, H36.TOP_NETS, True)
    R['datasets']['top']['wta_check'] = wc; R['datasets']['top']['overlap_with_h36'] = overlap('top', S['idx'])
    del TM, S
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    # QG
    QS = {}
    for gen in ('py', 'hw'):
        QS[gen] = H36.load_qg(gen); log(f'{gen}: {len(QS[gen]["y"])} jets; axes {QS[gen]["axinfo"]}')
    M = H36.load_qg_models()
    for name, src, seed in (('part_kin_perm1', 'part_kin', 1), ('pnet_kin_perm', 'pnet_kin', 0), ('pfn_k_perm', 'pfn_k', 0), ('efn_k_perm', 'efn_k', 0)):
        kind, arch, pidfeat, net = M.m[src]; M.m[name] = (kind, arch, pidfeat, permute_torch(net, seed).eval().to(M.dev))
    for gen in ('py', 'hw'):
        S = QS[gen]; REF = H36.QGRefs(gen); Jt = S['Jt']
        L = [np.c_[Jt['pt'][j, :Jt['n'][j]], Jt['eta'][j, :Jt['n'][j]], Jt['phi'][j, :Jt['n'][j]]] for j in range(len(Jt['n']))]
        wc = wta_check(S, [a[:, 1] for a in L], [a[:, 2] for a in L], [a[:, 0] for a in L], 100 if SMOKE else 300); log(f'{gen} WTA check', wc)

        def qg_scorer(rows, cond, S=S, REF=REF):
            return H36.qg_score(M, REF, H36.sub_jt(S['Jt'], rows), cond)
        R['datasets'][gen] = measure(gen, S, qg_scorer, H36.QG_NETS + ['LR_girth', 'LR_N'], False)
        R['datasets'][gen]['wta_check'] = wc; R['datasets'][gen]['overlap_with_h36'] = overlap(gen, S['idx'])
    # Holm per contrast across model x sample x axis
    cells = [(ds, nm, a) for ds in R['datasets'] for nm in R['datasets'][ds]['models'] for a in AXES]
    for con in ('AX-R1', 'AX-R2', 'AX-ANTI', 'AX-ROT', 'R1-R2', 'ANTI-R1'):
        ps = [R['datasets'][ds]['models'][nm]['axes'][a]['contrasts'][con]['mcnemar_p'] for ds, nm, a in cells]
        for (ds, nm, a), ph in zip(cells, H4._holm(ps)): R['datasets'][ds]['models'][nm]['axes'][a]['contrasts'][con]['p_holm'] = float(ph)
    R['holm_family'] = f'{len(cells)} cells per contrast'
    out = RES / ('h36_verify_smoke.json' if SMOKE else 'h36_verify.json'); out.write_text(json.dumps(R, indent=1, default=float)); log('wrote', out)


if __name__ == '__main__':
    main()
