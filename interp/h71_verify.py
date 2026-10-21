"""H71 verification (loop D). Re-runs the key JetClass part of the H71 energy-scale panel on a fresh sample and adds two controls.

--part jetclass (GPU): h1_irc_panel.jc_sample(2000, seed 7192) random rows per class (H71 used seed 7102); 1,000 per class
  evaluated, 1,000 per class train a reference MLP(log m, log pT) (rng 7202). Models ParT_full, ParT_kin, PN_full, PN_kin.
  Edits (constituent four-momenta, stored axis, H71 conventions):
    U(lam)   every p_i -> lam p_i                          (m x lam, pT x lam, m/pT fixed)
    UB(lam)  U(lam), then boost along the jet by 1/lam      (m x lam, pT fixed)
    B(lam)   boost along the jet only, |P| -> lam |P|        (m fixed, pT x lam): NEW control. It isolates the pT part of U;
             to first order in log lam, Delta s_U = Delta s_UB + Delta s_B for any smooth tagger.
    Ch / Nh / Rh / Rc at lam 0.9, 1.1 (charged-only, neutral-only, pT-share-matched random subset and its complement).
  lam in (0.9, 0.95, 1.05, 1.1) for U, UB, B.
--part stats (CPU): rho = sum_k size_U,k / sum_k size_UB,k (size = (|mean Ds(a)| + |mean Ds(b)|) / 2 over jets of class k,
  lam pair (a, b)) on all jets and on the jets whose edited pT stays inside the dataset's pT range at both a and b (NEW control
  for the training-range confound), for the fresh JetClass sample and for H71's own npz files (JetClass, Top, QG; read only).
  beta_B = sum_k size_B / sum_k size_UB (pT-at-fixed-mass part). Bootstrap 95 % CI over jets (stratified by class), sign-flip p.
Outputs: $AI_RESULTS/physics/h71_verify_jetclass.{json,npz}, h71_verify.json.
"""
import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import h12_boost_panel as H  # noqa: E402
from interp import h71_energy_scale_panel as P  # noqa: E402

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results'))
OUT = RES / 'physics'
LAMS = (0.9, 0.95, 1.05, 1.1)
SEED = 7192
NB = 1000
RANGE = dict(jetclass=(500.0, 1000.0), top=(550.0, 650.0), qg=(500.0, 550.0))
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def part_jetclass():
    from interp.h1_irc_panel import jc_sample, jc_table, pn_logits
    from interp.part_core import ParTNetwork, CL, inputs, build, sub
    from interp.p1v_verify_part import fwd
    from interp.pn_net import load_particlenet
    from jetdistill.config import MODELS
    dev = os.environ.get('JETDISTILL_DEVICE', 'cpu')
    Jall = jc_sample(2000, SEED); yall = Jall['label'].astype(int); rr = np.random.default_rng(SEED + 10); ev, rf = [], []
    for c in range(len(CL)):
        ii = rr.permutation(np.flatnonzero(yall == c)); ev.append(ii[:1000]); rf.append(ii[1000:])
    ev, rf = np.sort(np.concatenate(ev)), np.sort(np.concatenate(rf))
    J0 = sub(Jall, ev); y = yall[ev]
    tab = lambda Jx: (lambda Tj: dict(pt=Tj['pt'], eta=Tj['eta'], phi=Tj['phi'], m=Tj['m'], ax_eta=Tj['ax_eta'], ax_phi=Tj['ax_phi'], ext=Tj['ext']))(jc_table(Jx))
    T = tab(J0); Tp = tab(sub(Jall, rf)); Tp['ext'] = {}; yp = yall[rf]; del Jall
    log(f'JetClass verify: {len(ev)} evaluated + {len(rf)} reference jets; label counts {np.bincount(y).tolist()}')

    def prep(U):
        Jx = dict(J0); Jx['jet_eta'] = U['ax_eta'].astype(np.float32); Jx['jet_phi'] = U['ax_phi'].astype(np.float32)
        sg = np.where(Jx['jet_eta'] < 0, -1.0, 1.0)[:, None]; de, dp = H.rel(U)
        return build(Jx, U['pt'], sg * de, dp, U['m'], extra=U['ext'])
    pt_f = lambda net, n: (lambda J: fwd(net, *inputs(J, n))[1].astype(np.float64))
    models = {}
    for n in ('full', 'kin'):
        models[f'ParT_{n}'] = pt_f(ParTNetwork(n), n)
    for n, dims in (('full', 17), ('kin', 7)):
        pn = load_particlenet(Path(MODELS).parent / 'particlenet' / f'ParticleNet_{n}.pt', dims, dev)
        models[f'PN_{n}'] = (lambda J, p=pn, n=n: pn_logits(p, J, n))
    # reference MLP(log m, log pT), trained on the disjoint reference jets
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    fx = lambda O: np.stack([np.log(np.maximum(O['m'], 1e-3)), np.log(O['jpt'])], 1)
    Op = P.basic_obs(Tp); mlp = make_pipeline(StandardScaler(), MLPClassifier((32, 32), max_iter=400, early_stopping=True, random_state=0))
    mlp.fit(fx(Op), yp); models['MLP(m,pT)'] = ('obs', lambda O: np.log(np.clip(mlp.predict_proba(fx(O)), 1e-12, 1)))
    # edits
    real = T['pt'] > 0; ch = (T['ext']['charge'] != 0) & real; nh = real & ~ch
    rh, fch = P.matched_subset(T, ch, np.random.default_rng(SEED + 1)); rc = real & ~rh
    E = {}
    for lam in LAMS:
        E[f'U{lam}'] = P.scale_mask(T, lam, real)
        E[f'UB{lam}'] = H.boost(P.scale_mask(T, lam, real), 1.0 / lam)[0]
        E[f'B{lam}'] = H.boost(T, lam)[0]
    for lam in (0.9, 1.1):
        E[f'Ch{lam}'] = P.scale_mask(T, lam, ch); E[f'Nh{lam}'] = P.scale_mask(T, lam, nh)
        E[f'Rh{lam}'] = P.scale_mask(T, lam, rh); E[f'Rc{lam}'] = P.scale_mask(T, lam, rc)
    r = np.arange(len(y)); NPZ = dict(y=y.astype(np.int8)); res = dict(N=len(y), seed=SEED, classes=list(CL), models={}, edits={})
    O0 = P.basic_obs(T); NPZ['m__id'] = O0['m']; NPZ['jpt__id'] = O0['jpt']
    X0 = prep(H.sortT(T))
    for mn, f in models.items():
        lg = f[1](O0) if isinstance(f, tuple) else f(X0); lo = P.logodds_all(lg); NPZ[f'lo__{mn}__id'] = lo[r, y]
        res['models'][mn] = dict(acc=float((lg.argmax(1) == y).mean()))
        log(f'{mn}: accuracy {res["models"][mn]["acc"]:.4f} on N {len(y)}')
    for name, U in E.items():
        t = time.time(); O1 = P.basic_obs(U); X1 = prep(H.sortT(U))
        NPZ[f'm__{name}'] = O1['m']; NPZ[f'jpt__{name}'] = O1['jpt']
        for mn, f in models.items():
            lg = f[1](O1) if isinstance(f, tuple) else f(X1); NPZ[f'lo__{mn}__{name}'] = P.logodds_all(lg)[r, y]
        res['edits'][name] = dict(dlogm_median=float(np.median(np.log(np.maximum(O1['m'], 1e-6) / np.maximum(O0['m'], 1e-6)))),
                                  dlogpt_median=float(np.median(np.log(O1['jpt'] / O0['jpt']))))
        log(f'{name}: dlog m {res["edits"][name]["dlogm_median"]:+.4f} dlog pT {res["edits"][name]["dlogpt_median"]:+.4f} | '
            + ' '.join(f'{mn} {np.mean(NPZ[f"lo__{mn}__{name}"] - NPZ[f"lo__{mn}__id"]):+.3f}' for mn in models) + f' ({time.time() - t:.0f} s)')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'h71_verify_jetclass.json').write_text(json.dumps(res, indent=1))
    np.savez_compressed(OUT / 'h71_verify_jetclass.npz', **{k: np.asarray(v, np.float32) if k != 'y' else v for k, v in NPZ.items()})
    log('wrote h71_verify_jetclass.{json,npz}')


# ------------------------------------------------------------------------------------------------------------ statistics
def size_ratio(D, mn, y, sel, num_arm, den_arm, a, b, rng, C):
    """sum_k (|mean d_num(a)| + |mean d_num(b)|)/2 / sum_k same for den, over jets in sel; stratified bootstrap CI"""
    base = D[f'lo__{mn}__id'].astype(np.float64)
    d = lambda arm, l: D[f'lo__{mn}__{arm}{l}'].astype(np.float64) - base
    num = den = 0.0; bn, bd = np.zeros(NB), np.zeros(NB); n_used = 0
    for k in range(C):
        s = np.flatnonzero(sel & (y == k))
        if len(s) < 20: continue
        n_used += len(s)
        na, nb_, da, db = d(num_arm, a)[s], d(num_arm, b)[s], d(den_arm, a)[s], d(den_arm, b)[s]
        num += (abs(na.mean()) + abs(nb_.mean())) / 2; den += (abs(da.mean()) + abs(db.mean())) / 2
        I = rng.integers(0, len(s), (NB, len(s)))
        bn += (np.abs(na[I].mean(1)) + np.abs(nb_[I].mean(1))) / 2; bd += (np.abs(da[I].mean(1)) + np.abs(db[I].mean(1))) / 2
    if den == 0: return None, None
    return dict(value=float(num / den), ci=np.percentile(bn / np.maximum(bd, 1e-12), [2.5, 97.5]).tolist(), n=n_used), bn / np.maximum(bd, 1e-12)


def inrange(D, ds, a, b):
    lo, hi = RANGE[ds]; p = D['jpt__id'].astype(np.float64); return (p * a >= lo) & (p * a <= hi) & (p * b >= lo) & (p * b <= hi)


def analyse(D, ds, models, pairs, rng, arms=('U',)):
    y = D['y'].astype(int); C = int(y.max()) + 1; out = {}
    for mn in models:
        if f'lo__{mn}__id' not in D: continue
        o = {}
        for a, b in pairs:
            if f'lo__{mn}__U{a}' not in D or f'lo__{mn}__U{b}' not in D: continue
            sel_all = np.ones(len(y), bool); sel_in = inrange(D, ds, a, b); key = f'{a}/{b}'
            r_all, b_all = size_ratio(D, mn, y, sel_all, 'U', 'UB', a, b, rng, C)
            r_in, b_in = size_ratio(D, mn, y, sel_in, 'U', 'UB', a, b, rng, C)
            e = dict(frac_in_range=float(sel_in.mean()), rho_all=r_all, rho_in=r_in)
            if r_all and r_in:
                diff = b_in - b_all; e['rho_in_minus_all'] = dict(value=r_in['value'] - r_all['value'], ci=np.percentile(diff, [2.5, 97.5]).tolist(),
                                                                  p_boot=float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))))
            if f'lo__{mn}__B{a}' in D:
                for nm, sel in (('all', sel_all), ('in', sel_in)):
                    rb, _ = size_ratio(D, mn, y, sel, 'B', 'UB', a, b, rng, C); e[f'beta_B_{nm}'] = rb
                base = D[f'lo__{mn}__id'].astype(np.float64)
                dU, dUB, dB = (D[f'lo__{mn}__{x}{b}'].astype(np.float64) - base for x in ('U', 'UB', 'B'))
                e['additivity_b'] = dict(mean_abs_U=float(np.abs(dU).mean()), mean_abs_resid=float(np.abs(dU - dUB - dB).mean()))
            o[key] = e
        out[mn] = o
        g = lambda v, k: (v.get(k) or {}).get('value', float('nan'))
        log(f'{ds} {mn}: ' + ' | '.join(f'{k}: in-range {v["frac_in_range"]:.2f} rho_all {g(v, "rho_all"):.3f} rho_in {g(v, "rho_in"):.3f}'
                                        + (f' betaB_all {g(v, "beta_B_all"):.3f} betaB_in {g(v, "beta_B_in"):.3f}' if 'beta_B_in' in v else '')
                                        for k, v in o.items()))
    return out


def per_class_tests(D, mn, y, classes, rng, sel):
    """per class at lam 0.9 / 1.1 on jets in sel: mean Ds_U, Ds_UB, Ds_B, Ds_U - Ds_UB with CI and sign-flip p; asym excess at 1.1"""
    base = D[f'lo__{mn}__id'].astype(np.float64); out = {}
    for k, cn in enumerate(classes):
        s = sel & (y == k); d = lambda e: (D[f'lo__{mn}__{e}'].astype(np.float64) - base)[s]; c = {}
        for l in ('0.9', '1.1'):
            for arm in ('U', 'UB', 'B'):
                if f'lo__{mn}__{arm}{l}' in D: c[f'{arm}{l}'] = P.mstat(d(f'{arm}{l}'), rng)
            c[f'U-UB{l}'] = P.mstat(d(f'U{l}') - d(f'UB{l}'), rng)
        if f'lo__{mn}__Ch1.1' in D:
            c['asym_excess1.1'] = P.mstat((d('Ch1.1') - d('Nh1.1')) - (d('Rh1.1') - d('Rc1.1')), rng)
        c['n'] = int(s.sum()); out[cn] = c
    return out


def part_stats():
    rng = np.random.default_rng(7299); S = dict(ranges=RANGE, fresh={}, h71={}, holm={})
    nets = ('ParT_full', 'ParT_kin', 'PN_full', 'PN_kin', 'MLP(m,pT)')
    fj = OUT / 'h71_verify_jetclass.npz'
    if fj.exists():
        D = dict(np.load(fj)); R = json.loads((OUT / 'h71_verify_jetclass.json').read_text()); y = D['y'].astype(int)
        S['fresh']['accuracy'] = R['models']; S['fresh']['N'] = R['N']; S['fresh']['seed'] = R['seed']
        S['fresh']['rho'] = analyse(D, 'jetclass', nets, ((0.9, 1.1), (0.95, 1.05)), rng)
        sel_in = inrange(D, 'jetclass', 0.9, 1.1); S['fresh']['tests'] = {}; tests = dict(U_eq_UB_in=[], U_eq_UB_all=[], asym=[], B_eq0_in=[])
        for mn in nets[:4]:
            S['fresh']['tests'][mn] = dict(all=per_class_tests(D, mn, y, R['classes'], rng, np.ones(len(y), bool)),
                                           inrange=per_class_tests(D, mn, y, R['classes'], rng, sel_in))
            for cn in R['classes']:
                for l in ('0.9', '1.1'):
                    tests['U_eq_UB_in'].append((f'{mn}|{cn}|{l}', S['fresh']['tests'][mn]['inrange'][cn][f'U-UB{l}']['p_signflip_eq0']))
                    tests['U_eq_UB_all'].append((f'{mn}|{cn}|{l}', S['fresh']['tests'][mn]['all'][cn][f'U-UB{l}']['p_signflip_eq0']))
                    tests['B_eq0_in'].append((f'{mn}|{cn}|{l}', S['fresh']['tests'][mn]['inrange'][cn][f'B{l}']['p_signflip_eq0']))
                tests['asym'].append((f'{mn}|{cn}', S['fresh']['tests'][mn]['all'][cn]['asym_excess1.1']['p_signflip_eq0']))
        for fam, lst in tests.items():
            hp = P.holm([x[1] for x in lst]); S['holm'][fam] = dict(n=len(lst), n_below_0p05=int(sum(h < 0.05 for h in hp)), p=dict(zip([x[0] for x in lst], hp)))
            log(f'Holm {fam}: {S["holm"][fam]["n_below_0p05"]} / {len(lst)}')
            if fam == 'asym':
                for mn in nets[:4]:
                    log(f'  asym excess 1.1 {mn}: ' + ' '.join(f'{cn} {S["fresh"]["tests"][mn]["all"][cn]["asym_excess1.1"]["mean"]:+.3f}(H {S["holm"][fam]["p"][f"{mn}|{cn}"]:.1e})' for cn in R['classes']))
    for ds, mods in (('jetclass', nets), ('top', ('ParT_kin', 'PN_kin', 'LorentzNet', 'MLP(m,pT)')),
                     ('qg', ('part_kin', 'pnet_kin', 'part_kinpid', 'pnet_kinpid', 'lorentznet', 'MLP(m,pT)'))):
        fn = OUT / f'h71_{ds}.npz'
        if not fn.exists(): continue
        D = dict(np.load(fn)); S['h71'][ds] = analyse(D, ds, mods, ((0.9, 1.1), (0.95, 1.05), (0.98, 1.02)), rng)
    (OUT / 'h71_verify.json').write_text(json.dumps(S, indent=1, default=float)); log('wrote h71_verify.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--part', required=True, choices=['jetclass', 'stats'])
    a = ap.parse_args(); log(f'H71 verify part {a.part}')
    dict(jetclass=part_jetclass, stats=part_stats)[a.part](); log('done')
