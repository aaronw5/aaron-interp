"""H49 verification (loop D, verifier): fresh-sample rerun of interp.h49_groomed_mass plus two checks the H49 run lacks.

1. Fresh samples: seeds hls 4931, jc 4932, top 4933 (H49 used 4901-4903); everything else as in H49 (same code, imported).
2. Per-jet direction test of the 'groomed-like' extra response. For an m_SD follower the sign of D = s(G_in) - s(G_out)
   must follow the sign of the MLP(m_SD) prediction jet by jet: W jets whose m_SD sits below the W peak gain score when
   G_in raises m_SD, jets above it lose score. A response to 'soft particles next to a hard prong' (any mechanism other
   than m_SD) predicts D < 0 in both groups. Groups: D_ref = D of MLP(m_SD) > +0.05 ('rise') or < -0.05 ('fall');
   plus Spearman rho(D_net, D_ref) with a permutation p, and bins of the unedited m_SD.
   The same test on the shared response (G_out, groomed away, raises the plain mass m): groups by the MLP(m) prediction
   Delta s_MLP(m)(G_out) > +0.05 or < -0.05; a plain-mass follower gains score in the 'rise' group.
3. Control G_dense: the G_out particles moved to the G_out candidate direction (same circle r*, same >= 1.1 R_g from both
   prongs, so soft drop still removes them) with the MOST unedited pT within 0.5 R_g instead of the least. Same count,
   pT, radius (so same ungroomed Delta m to leading order) and Delta m_SD = 0: it isolates the effect of placing the soft
   particles next to existing radiation at fixed m and m_SD. Compared with D.
Outputs: $AI_RESULTS/physics/h49_verify_{hls,gpu}.json (+ h49_verify_groomed_{hls,gpu}.png, the H49 figure on the fresh
sample); intermediate npz in /tmp of the pod only.
  ./part_job.sh d-h49-verify 'python -m interp.h49_verify --part gpu'
  ./job.sh d-h49-vhls 1 4 'python -m interp.h49_verify --part hls'
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import interp.h49_groomed_mass as H  # noqa: E402
from interp.h1_irc_panel import finalize, wrap  # noqa: E402
from interp.h18_softcount_cross import eaxis  # noqa: E402

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'physics'
TMP = Path('/tmp/h49v')
H.OUT = TMP
H.SEEDS.update(hls=4931, jc=4932, top=4933)
log = H.log


def boot_ci(x, B=2000, seed=0):
    x = np.asarray(x, np.float64)
    if len(x) < 5: return None
    i = np.random.default_rng(seed).integers(0, len(x), (B, len(x))); m = x[i].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def signflip_p(d, P=5000, seed=1):
    d = np.asarray(d, np.float64)
    if len(d) < 5: return None
    rng = np.random.default_rng(seed); obs = abs(d.mean())
    s = rng.choice([-1.0, 1.0], (P, len(d))); return float(((np.abs((s * d).mean(1)) >= obs - 1e-15).sum() + 1) / (P + 1))


def spearman_perm(a, b, P=2000, seed=2):
    from scipy.stats import rankdata
    a, b = rankdata(a), rankdata(b); a = (a - a.mean()) / a.std(); b = (b - b.mean()) / b.std(); r = float((a * b).mean())
    rng = np.random.default_rng(seed); null = np.array([(a[rng.permutation(len(a))] * b).mean() for _ in range(P)])
    return r, float(((np.abs(null) >= abs(r) - 1e-15).sum() + 1) / (P + 1))


# ------------------------------------------------------------------------------------------------ G_dense control
def dense_edit(T0, cap, G):
    """G_out particles at the qualifying candidate direction with the most unedited pT within 0.5 R_g"""
    ae, ap, Rg, rstar, dF, dN, q, rF = G['ae'], G['ap'], G['Rg'], G['rstar'], G['dF'], G['dN'], G['q'], G['rF']
    thF = np.where(rF > 1e-3, np.arctan2(dF[:, 1], dF[:, 0]), np.arctan2(dF[:, 1] - dN[:, 1], dF[:, 0] - dN[:, 0]))
    real = T0['pt'] > 0; de = T0['eta'] - ae[:, None]; dp = wrap(T0['phi'] - ap[:, None])
    cand = np.deg2rad(np.arange(60, 301, 30)); cost_min, cost_max, npts, ok = [], [], [], []
    for c in cand:
        th = thF + c; ce, cp = rstar * np.cos(th), rstar * np.sin(th)
        near = real & (np.hypot(de - ce[:, None], dp - cp[:, None]) < 0.5 * Rg[:, None]); npt = (T0['pt'] * near).sum(1)
        dmin = np.minimum(np.hypot(ce - dF[:, 0], cp - dF[:, 1]), np.hypot(ce - dN[:, 0], cp - dN[:, 1]))
        q_ = dmin >= 1.1 * Rg; npts.append(npt); ok.append(q_)
        cost_min.append(np.where(q_, npt, 1e6 + npt - dmin)); cost_max.append(np.where(q_, -npt, 1e6 - dmin))
    npts = np.stack(npts, 1); ok = np.stack(ok, 1)
    iO = np.argmin(np.stack(cost_min, 1), 1); iD = np.argmin(np.stack(cost_max, 1), 1)
    thD = thF + cand[iD]; dth = H.SPACE * Rg / np.maximum(rstar, 1e-6); offs = np.array([-1.0, 0.0, 1.0])
    ang = thD[:, None] + offs[None] * dth[:, None]; Q = np.repeat(q[:, None], H.K, 1)
    e_, p_ = ae[:, None] + rstar[:, None] * np.cos(ang), ap[:, None] + rstar[:, None] * np.sin(ang)
    Ud, _ = finalize(H.add_parts(T0, Q, e_, p_), cap)
    rows = np.arange(len(q))
    # unedited pT within 0.5 R_g of the G_in centre (direction of F at radius r_in), for comparison
    ci_, cpi = G['rin'] * np.cos(thF), G['rin'] * np.sin(thF)
    nin = (T0['pt'] * (real & (np.hypot(de - ci_[:, None], dp - cpi[:, None]) < 0.5 * Rg[:, None]))).sum(1)
    info = dict(npt_out=npts[rows, iO], npt_dense=npts[rows, iD], npt_in=nin, n_ok=ok.sum(1), same_dir=(iO == iD))
    return Ud, info


# ------------------------------------------------------------------------------------------------ one dataset
def run(ds, T0, y, scorer, info):
    H.SEEDS['_cur'] = H.SEEDS[ds]
    conds, G, valid = H.run_dataset(ds, T0, y, scorer, 0, info)
    Ud, dinf = dense_edit(T0, H.CAP[ds], G)
    Od = H.observables({'Gd': Ud}, G['ae'], G['ap'], do_tau=False)
    Z = H.load_ds(ds, 0)
    S_rc = scorer(Ud, *eaxis(Ud)); S_st = scorer(Ud, T0['ax_eta'], T0['ax_phi'])
    out = dict(seed=H.SEEDS[ds], n=int(len(y)), dense={}, direction={}, msd_bins={})
    dm_d = Od['m|Gd'] - Z['m|base']; dsd_d = Od['msd|Gd'] - Z['msd|base']; dm_o = Z['m|Gout'] - Z['m|base']
    vd = valid & (np.abs(dsd_d) < 0.5) & (np.abs(dm_d - dm_o) < 1.0) & ~dinf['same_dir']
    out['dense_geometry'] = {}
    tests = []
    for cls, k in H.SIG[ds].items():
        sig = valid & (y == k); sd_ = vd & (y == k)
        out['dense_geometry'][cls] = dict(n_matched=int(sig.sum()), n_dense=int(sd_.sum()),
                                          npt_out_med=float(np.median(dinf['npt_out'][sd_])) if sd_.any() else None,
                                          npt_dense_med=float(np.median(dinf['npt_dense'][sd_])) if sd_.any() else None,
                                          npt_in_med=float(np.median(dinf['npt_in'][sd_])) if sd_.any() else None,
                                          dm_dense_med=float(np.median(dm_d[sd_])) if sd_.any() else None,
                                          dm_out_med=float(np.median(dm_o[sd_])) if sd_.any() else None,
                                          dmsd_dense_med=float(np.median(dsd_d[sd_])) if sd_.any() else None)
        Dr = (Z[f's|MLPmsd|{cls}|Gin_st'] - Z[f's|MLPmsd|{cls}|Gout_st']).astype(np.float64)
        Gm = (Z[f's|MLPm|{cls}|Gout_st'] - Z[f's|MLPm|{cls}|base_st']).astype(np.float64)
        msd0 = Z['msd|base']
        models = sorted({kk.split('|')[1] for kk in Z if kk.startswith('s|') and kk.endswith(f'|{cls}|base_rc') and not kk.split('|')[1].startswith('MLP')})
        for mn in models + ['MLPmsd', 'MLPm']:
            b = 'st' if mn.startswith('MLP') else 'rc'
            s = lambda c: Z[f's|{mn}|{cls}|{c}_{b}'].astype(np.float64)
            D = s('Gin') - s('Gout'); Go = s('Gout') - s('base')
            key = f'{mn}|{cls}'; r = {}
            for gname, gm in (('rise', Dr > 0.05), ('fall', Dr < -0.05)):
                rr = sig & gm
                r[f'D_{gname}'] = dict(n=int(rr.sum()), mean=float(D[rr].mean()) if rr.any() else None, ci=boot_ci(D[rr], seed=3),
                                       p=signflip_p(D[rr]), ref_mean=float(Dr[rr].mean()) if rr.any() else None)
                if not mn.startswith('MLP') and r[f'D_{gname}']['p'] is not None:
                    tests.append((ds, key, f'D_{gname}', r[f'D_{gname}']['p']))
            rho, pr = spearman_perm(D[sig], Dr[sig]); r['spearman_D_vs_MLPmsd'] = dict(rho=rho, p=pr)
            for gname, gm in (('rise', Gm > 0.05), ('fall', Gm < -0.05)):
                rr = sig & gm
                r[f'Gout_{gname}'] = dict(n=int(rr.sum()), mean=float(Go[rr].mean()) if rr.any() else None, ci=boot_ci(Go[rr], seed=4),
                                          ref_mean=float(Gm[rr].mean()) if rr.any() else None)
            rho2, pr2 = spearman_perm(Go[sig], Gm[sig]); r['spearman_Gout_vs_MLPm'] = dict(rho=rho2, p=pr2)
            out['direction'][key] = r
            # bins of the unedited groomed mass
            edges = [0, 50, 65, 75, 85, 95, 110, 1e9] if cls in ('W', 'Z') else [0, 120, 150, 165, 180, 200, 1e9]
            bins = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                rr = sig & (msd0 >= lo) & (msd0 < hi)
                if rr.sum() >= 20:
                    bins.append(dict(lo=lo, hi=hi, n=int(rr.sum()), D=float(D[rr].mean()), D_ci=boot_ci(D[rr], seed=5),
                                     D_ref=float(Dr[rr].mean()), Gout=float(Go[rr].mean()), Gout_ref_MLPm=float(Gm[rr].mean())))
            out['msd_bins'][key] = bins
            # dense control: Delta s(G_dense) - Delta s(G_out), groomed-away both
            if not mn.startswith('MLP') and mn in S_rc:
                Ld = np.asarray(S_rc[mn] if b == 'rc' else S_st[mn], np.float64)
                sdn = H.class_scores(Ld, ds)[cls]
                Dd = sdn - s('Gout'); rr = sd_
                out['dense'][key] = dict(n=int(rr.sum()), Ddense=float(Dd[rr].mean()) if rr.any() else None, Ddense_ci=boot_ci(Dd[rr], seed=6),
                                         p=signflip_p(Dd[rr]), D_same_jets=float(D[rr].mean()) if rr.any() else None,
                                         D_same_jets_ci=boot_ci(D[rr], seed=6), ds_dense=float((sdn - s('base'))[rr].mean()) if rr.any() else None,
                                         ds_out=float(Go[rr].mean()) if rr.any() else None)
            log(f'{ds} {key}: D rise {r["D_rise"]["mean"]} (ref {r["D_rise"]["ref_mean"]}) fall {r["D_fall"]["mean"]} (ref {r["D_fall"]["ref_mean"]}); '
                f'rho {rho:.3f}; Gout rise {r["Gout_rise"]["mean"]} fall {r["Gout_fall"]["mean"]}; dense {out["dense"].get(key)}')
    return out, tests


def fresh_summary(tag):
    H.part_summary(0)
    R = json.loads((TMP / 'h49_summary.json').read_text()); keep = {}
    for ds, d in R['datasets'].items():
        keep[ds] = dict(unedited=d.get('info', {}).get('unedited'), matched_fraction=d.get('info', {}).get('matched_fraction', {}).get('ALL'),
                        geometry=d.get('info', {}).get('geometry'), axis_offset_mrad_median=d.get('axis_offset_mrad_median'),
                        windows={k: dict(D_effS=w['D_effS'], D_effS_ci=w['D_effS_ci'], effS=w['effS']) for k, w in d['windows'].items()},
                        models={mn: {k: {f: st.get(f) for f in ('n_sig', 'D', 'D_ci', 'D_p', 'ratio_to_MLPmsd', 'ratio_to_MLPmsd_ci', 'ds_Gin', 'ds_Gout',
                                                                 'ds_c0', 'ds_rand', 'D_minus_TN', 'D_qcd', 'effS_in_minus_out', 'effS_in_minus_out_ci',
                                                                 'mcnemar_in_vs_out')} for k, st in m.items()} for mn, m in d['models'].items()})
    keep['tests_holm'] = R['tests']
    if (TMP / 'h49_groomed.png').exists(): shutil.copy(TMP / 'h49_groomed.png', RES / f'h49_verify_groomed_{tag}.png')
    return keep


def holm_add(tests):
    from interp.h1_irc_panel import holm
    if not tests: return []
    h = holm([t[3] for t in tests]); return [dict(ds=a, key=b, test=c, p=p, holm=float(hv)) for (a, b, c, p), hv in zip(tests, h)]


def part_gpu():
    import torch
    from interp.part_core import ParTNetwork, inputs
    from interp.p1v_verify_part import fwd
    from interp.pn_net import load_particlenet
    from interp.h1_irc_panel import pn_logits, jc_table
    from interp.h13_pileup_panel import jc_maker, TopModels, top_J, top_table
    from interp import data_top as dt
    from jetdistill.config import MODELS
    dev = os.environ.get('JETDISTILL_DEVICE', 'cpu'); res = {}; tests = []
    # JetClass
    J0 = H.jc_sample4(2000, H.SEEDS['jc']); y = J0['label'].astype(int)
    T0 = finalize(jc_table(J0), 128)[0]; make = jc_maker(J0); nets = {}
    for n in ('full', 'kin'):
        net = ParTNetwork(n); nets[f'ParT_{n}'] = (lambda J, net=net, n=n: np.asarray(fwd(net, *inputs(J, n))[1], np.float64))
    for n, dims in (('full', 17), ('kin', 7)):
        pn = load_particlenet(Path(MODELS).parent / 'particlenet' / f'ParticleNet_{n}.pt', dims, dev)
        nets[f'PN_{n}'] = (lambda J, pn=pn, n=n: pn_logits(pn, J, n))
    scorer = lambda Uf, ae, ap: {mn: f(make(Uf, ae, ap)) for mn, f in nets.items()}
    res['jc'], t = run('jc', T0, y, scorer, dict(sample=f'JetClass 2000 random rows per class (QCD, Zqq, Wqq, Tbqq), seed {H.SEEDS["jc"]}'))
    tests += t
    # Top Landscape
    S = dt.sample(3000, H.SEEDS['top']); y = S['label'].astype(int); T0 = finalize(top_table(S['full']), 200)[0]
    Mo = TopModels('cuda' if torch.cuda.is_available() else 'cpu'); order = np.random.default_rng(H.SEEDS['top'] + 7).permutation(len(y))
    scorer = lambda Uf, ae, ap: Mo.score(top_J(Uf, ae, ap), order)
    res['top'], t = run('top', T0, y, scorer, dict(sample=f'Top Landscape test, data_top.sample(3000, {H.SEEDS["top"]})', idx=S['idx'].tolist()))
    tests += t
    res['fresh_summary'] = fresh_summary('gpu'); res['holm_direction_tests'] = holm_add(tests)
    (RES / 'h49_verify_gpu.json').write_text(json.dumps(res, indent=1, default=float)); log('wrote h49_verify_gpu.json')


def part_hls():
    from interp.jedi_core import Jedi
    from interp.h1_irc_panel import hls_sample
    from interp.h13_pileup_panel import hls_in
    T0, y = hls_sample(2000, H.SEEDS['hls']); T0 = finalize(T0, 64)[0]
    net = Jedi(64, '3-feature')
    scorer = lambda Uf, ae, ap: {'JEDI': np.asarray(net.run(hls_in(Uf, ae, ap))['logits'], np.float64)}
    res = {}
    res['hls'], t = run('hls', T0, y, scorer, dict(sample=f'jedi_core.sample(2000, {H.SEEDS["hls"]}) jets', frac_at_64=float((H.nreal(T0) >= 64).mean())))
    res['fresh_summary'] = fresh_summary('hls'); res['holm_direction_tests'] = holm_add(t)
    (RES / 'h49_verify_hls.json').write_text(json.dumps(res, indent=1, default=float)); log('wrote h49_verify_hls.json')


if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('--part', required=True, choices=['gpu', 'hls']); A = a.parse_args()
    TMP.mkdir(parents=True, exist_ok=True); RES.mkdir(parents=True, exist_ok=True)
    dict(gpu=part_gpu, hls=part_hls)[A.part](); log('done')
