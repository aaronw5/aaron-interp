"""H54 verification (loop D): fresh-seed regeneration of the Hz+/Hz- hadronisation variants plus single-knob hadronisation
settings, and a dose-response control across all hadronisation settings (pilot + fresh).

Why: H54's Hz+/Hz- were chosen so that the KMT-trained EFN output does not move. That zero is a cancellation between the
a/b knobs (EFN toward gluon) and the sigma knob (EFN toward quark). If the networks read the same IRC-safe information as
the EFN but weight the a/b-type and sigma-type IRC-safe changes differently, they would move at the chosen point without
reading anything beyond IRC-safe energy flow. Control: settings that change ONE knob with about the same |dN/N| (~9-10 %)
but EFN shifts of opposite sign. Under "network = IRC-safe like the EFN" the network shift follows the sign of the EFN shift;
under "network also reads multiplicity / soft-hadron structure" it follows dN/N in every setting.

Steps
  --step gen    (CPU 8, env-gen) 7 settings x 8,000 q + 8,000 g jets, new per-event seeds (worker ids 6001-6004, not used
                by H54: main 5401-5408, pilot 5501-5502) -> data/qg/gen/verify/<setting>.npz
  --step score  (GPU, env-part) models + LR(N), LR(pT^D) + observables on the fresh settings and on the 25 pilot settings;
                statistics -> results/physics/h54_verify.json, h54_verify_scores.npz
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from interp import h54_qg_generator_knobs as G

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
V = Path('/j-jepa-vol/aaron-interp')
RES = Path(os.environ.get('AI_RESULTS', V / 'results')) / 'physics'
VG = V / 'data' / 'qg' / 'gen' / 'verify'
WBASE = 6001
WPC = 2                                   # workers per class
SETTINGS = {                               # (aLund, bLund, StringPT:sigma)
    'V0': (0.68, 0.98, 0.335),
    'Hzp': (0.79, 0.88, 0.247), 'Hzm': (0.48, 0.98, 0.438),           # H54's chosen settings
    'sig_Nup': (0.68, 0.98, 0.235), 'a_Nup': (1.28, 0.98, 0.335),       # one knob, N up (pilot: EFN + vs -)
    'sig_Ndn': (0.68, 0.98, 0.45), 'a_Ndn': (0.18, 0.98, 0.335),        # one knob, N down (pilot: EFN - vs +)
}
MODELS = ['part_kin', 'part_kinpid', 'pnet_kin', 'pnet_kinpid', 'lorentznet', 'pfn_k', 'efn_k', 'efn_t', 'hw_part_kin',
          'hw_pnet_kin', 'perm0', 'lr_n', 'lr_ptd']
NETS = ['part_kin', 'part_kinpid', 'pnet_kin', 'pnet_kinpid', 'lorentznet', 'pfn_k', 'hw_part_kin', 'hw_pnet_kin', 'perm0']


def step_gen(a):
    from multiprocessing import Pool
    VG.mkdir(parents=True, exist_ok=True); tmp = VG / 'tmp'; tmp.mkdir(exist_ok=True)
    tasks = []
    for name, (aa, bb, ss) in SETTINGS.items():
        extra = {} if name == 'V0' else {'StringZ:aLund': aa, 'StringZ:bLund': bb, 'StringPT:sigma': ss}
        for cls in (0, 1):
            for w in range(WPC):
                tasks.append(dict(cls=cls, widx=WBASE + cls * WPC + w, extra=extra, n_jets=a.jets // WPC, max_events=45_000,
                                  out=str(tmp / f'{name}_{cls}_{w}.npz'), hash_events=100, name=name))
    with Pool(a.cpu) as pool: summ = pool.map(G.gen_worker, tasks)
    info = {}
    for name in SETTINGS:
        tt = [t for t in tasks if t['name'] == name]; Z = [np.load(t['out']) for t in tt]
        Lmax = max(z['X'].shape[1] for z in Z)
        X = np.concatenate([np.pad(z['X'], ((0, 0), (0, Lmax - z['X'].shape[1]), (0, 0))) for z in Z])
        np.savez_compressed(VG / f'{name}.npz', X=X, n=np.concatenate([z['n'] for z in Z]), y=np.concatenate([z['y'] for z in Z]),
                            event=np.concatenate([z['event'] for z in Z]), widx=np.concatenate([z['widx'] for z in Z]),
                            jet=np.concatenate([z['jet'] for z in Z]), hashes=np.concatenate([z['hashes'] for z in Z]),
                            hash_widx=np.concatenate([np.full(len(z['hashes']), t['widx']) for z, t in zip(Z, tt)]))
        info[name] = dict(setting=SETTINGS[name], jets=int(len(X)), workers=[s for s, t in zip(summ, tasks) if t['name'] == name])
        for t in tt: os.remove(t['out'])
        log('wrote', name, len(X))
    Z0 = np.load(VG / 'V0.npz'); h0 = {(int(w), int(h[0])): (h[1], h[2]) for w, h in zip(Z0['hash_widx'], Z0['hashes'])}
    for name in SETTINGS:
        Z1 = np.load(VG / f'{name}.npz'); h1 = {(int(w), int(h[0])): (h[1], h[2]) for w, h in zip(Z1['hash_widx'], Z1['hashes'])}
        com = sorted(set(h0) & set(h1)); info[name]['parton_identical_frac'] = float(np.mean([h0[k] == h1[k] for k in com])) if com else None
        info[name]['parton_hash_n'] = len(com); log(name, 'parton level identical to V0 in', info[name]['parton_identical_frac'], 'of', len(com))
    (VG / 'gen_info.json').write_text(json.dumps(info, indent=1, default=float))


def _load(path):
    Z = np.load(path); X, n = Z['X'], Z['n']
    L = [X[j, :n[j]].astype(np.float64) for j in range(len(n))]
    return L, Z['y'].astype(int), Z['widx'].astype(np.int64) * 100_000 + Z['event'].astype(np.int64), Z


def step_score(a):
    import torch
    import copy
    from interp import h5_qg_counting as H
    from interp import h14_qg_generator as H14
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; RES.mkdir(parents=True, exist_ok=True)
    M = H.Models(dev)
    for name, arch, pf in (('part_kin', 'part', False), ('part_kinpid', 'part', True), ('pnet_kin', 'pn', False), ('pnet_kinpid', 'pn', True)):
        M.add_weaver(name, arch, pf)
    M.add_lnet()
    for kind in ('efn', 'pfn'): H14.add_keras_ef(M, kind)
    net = H.make_ef('efn'); net.load_state_dict(torch.load(V / 'models' / 'qg' / 'h5_efn_torch.pt', map_location='cpu'))
    M.add_torch('efn_t', net, json.loads((V / 'models' / 'qg' / 'h5_efn_torch.json').read_text()))
    for name, arch in (('part_kin', 'part'), ('pnet_kin', 'pn')):
        H14.add_weaver_dir(M, 'hw_' + name, arch, False, V / 'models' / 'qg_herwig' / name)
    kind, arch, pf, n0 = M.m['part_kin']; g = torch.Generator().manual_seed(0); n2 = copy.deepcopy(n0).cpu()
    with torch.no_grad():
        for prm in n2.parameters():
            fl = prm.data.view(-1); fl.copy_(fl[torch.randperm(fl.numel(), generator=g)])
    M.m['perm0'] = (kind, arch, pf, n2.eval().to(dev))
    T, _ = H14.lrn_tables(); lrn = -T['py']
    Lt, yt, _ = H14.efp_train_lists('py', n_per_class=30000, seed=G.SEED + 2)
    ptd_t = np.array([np.sqrt(((x[:, 0] / x[:, 0].sum()) ** 2).sum()) for x in Lt])
    bp = np.quantile(ptd_t, np.linspace(0, 1, 41)); bp[0], bp[-1] = 0, 1.01; hp = G._lr_hist(ptd_t, yt, bp)
    samples = {f'F|{k}': VG / f'{k}.npz' for k in SETTINGS}
    pil = G.GEN / 'pilot'; pnames = sorted({p.name.rsplit('_', 1)[0] for p in pil.glob('*_0.npz')})
    D = {}
    for k, p in samples.items(): D[k] = _load(p)
    for nm in pnames:                                         # pilot: per-class files, merged
        a0, a1 = _load(pil / f'{nm}_0.npz'), _load(pil / f'{nm}_1.npz')
        D[f'P|{nm}'] = (a0[0] + a1[0], np.r_[a0[1], a1[1]], np.r_[a0[2], a1[2]], None)
    out = {}
    for sname, (L, y, key, Z) in D.items():
        O = G.jet_obs(L); out[f'y|{sname}'] = y.astype(np.int8); out[f'key|{sname}'] = key
        for k in ('N', 'ptd', 'girth', 'm_over_pt', 'C1_1.0'): out[f'obs|{sname}|{k}'] = O[k].astype(np.float32)
        out[f'{sname}|lr_n'] = lrn[np.minimum(O['N'].astype(int), 299)].astype(np.float32)
        out[f'{sname}|lr_ptd'] = hp[np.clip(np.searchsorted(bp, O['ptd'], side='right') - 1, 0, 39)].astype(np.float32)
        for nm in [m for m in MODELS if m in M.m]:
            v = []
            for s in range(0, len(L), 10000):
                Ls = L[s:s + 10000]; v.append(-M.logit(nm, H.from_list(Ls, Lpad=max(len(x) for x in Ls))))
            out[f'{sname}|{nm}'] = np.concatenate(v).astype(np.float32)
        log(sname, len(L), 'jets; acc ' + ' '.join(f'{m} {np.mean((out[f"{sname}|{m}"] > 0) == (y == 1)):.3f}' for m in ('part_kin', 'efn_k', 'lr_n')))
    np.savez_compressed(RES / 'h54_verify_scores.npz', **out)
    stats(out, list(D), a.boot)


def stats(out, snames, B):
    from scipy.stats import norm
    from interp.h5_qg_counting import holm
    models = [m for m in MODELS if f'F|V0|{m}' in out]
    rng = np.random.default_rng(54054)
    R = dict(models=models, B=B, accuracy={}, fresh={}, pilot={}, regression={}, tests=[])
    for s in snames:
        y = out[f'y|{s}']
        R['accuracy'][s] = {m: float(np.mean((out[f'{s}|{m}'] > 0) == (y == 1))) for m in models}

    def paired(s0, s1, cls):
        y0, y1 = out[f'y|{s0}'], out[f'y|{s1}']; k0, k1 = out[f'key|{s0}'], out[f'key|{s1}']
        pos = {int(k): j for j, k in enumerate(k0) if y0[j] == cls}
        pr = [(pos[int(k)], j) for j, k in enumerate(k1) if y1[j] == cls and int(k) in pos]
        return np.array([p[0] for p in pr], int), np.array([p[1] for p in pr], int)

    def shift_block(s0, s1, cls, Bb):
        i0, i1 = paired(s0, s1, cls); J = len(i0)
        Dm = np.stack([out[f'{s1}|{m}'][i1].astype(np.float64) - out[f'{s0}|{m}'][i0] for m in models], 1)
        Do = {k: (out[f'obs|{s1}|{k}'][i1].astype(np.float64), out[f'obs|{s0}|{k}'][i0].astype(np.float64)) for k in ('N', 'ptd', 'girth', 'm_over_pt', 'C1_1.0')}
        bm = np.empty((Bb, len(models)))
        for b in range(Bb):
            c = np.bincount(rng.integers(0, J, J), minlength=J).astype(np.float64); bm[b] = c @ Dm / J
        r = dict(n_pairs=int(J), obs_rel={k: float(v1.mean() / v0.mean() - 1) for k, (v1, v0) in Do.items()})
        # unpaired (all jets of the class, as H54) for comparison
        y0, y1 = out[f'y|{s0}'], out[f'y|{s1}']
        r['unpaired'] = {m: float(out[f'{s1}|{m}'][y1 == cls].mean() - out[f'{s0}|{m}'][y0 == cls].mean()) for m in models}
        d = Dm.mean(0); mi = {m: i for i, m in enumerate(models)}
        for m in models:
            i = mi[m]; r[m] = dict(d=float(d[i]), ci=np.percentile(bm[:, i], [2.5, 97.5]).tolist(), se=float(bm[:, i].std()))
            if m not in ('efn_k', 'efn_t', 'lr_n', 'lr_ptd'):
                for ref in ('efn_k', 'lr_n'):
                    j = mi[ref]; dd = d[i] - d[j]; bd = bm[:, i] - bm[:, j]; se = bd.std()
                    r[m][f'minus_{ref}'] = dict(d=float(dd), ci=np.percentile(bd, [2.5, 97.5]).tolist(), p=float(2 * norm.sf(abs(dd) / se)) if se > 0 else 1.0)
                e, n_ = mi['efn_k'], mi['lr_n']; den = d[n_] - d[e]; bk = (bm[:, i] - bm[:, e]) / (bm[:, n_] - bm[:, e])
                r[m]['kappa_lr'] = dict(k=float((d[i] - d[e]) / den), ci=np.percentile(bk, [2.5, 97.5]).tolist(), den=float(den))
        return r

    for s in SETTINGS:
        if s == 'V0': continue
        for cls, cn in ((1, 'q'), (0, 'g')):
            r = shift_block('F|V0', f'F|{s}', cls, B); R['fresh'].setdefault(s, {})[cn] = r
            for m in NETS:
                if m == 'perm0': continue
                R['tests'].append(dict(sample='fresh', setting=s, cls=cn, model=m, ref='efn_k', d=r[m]['minus_efn_k']['d'], p=r[m]['minus_efn_k']['p']))
            log('fresh', s, cn, f"dN/N {100 * r['obs_rel']['N']:+.1f} % ptd {100 * r['obs_rel']['ptd']:+.1f} %", ' '.join(f"{m} {r[m]['d']:+.3f}" for m in models))
    for s in snames:
        if not s.startswith('P|') or s == 'P|nominal': continue
        for cls, cn in ((1, 'q'), (0, 'g')):
            R['pilot'].setdefault(s[2:], {})[cn] = shift_block('P|nominal', s, cls, 300)
    adj = holm([t['p'] for t in R['tests']])
    for t, h in zip(R['tests'], adj): t['holm'] = float(h)
    # dose-response across hadronisation settings (pilot 24 + fresh 6): delta_net = alpha * delta_EFN + beta * delta_LR(N)
    # [+ gamma * delta_LR(pT^D)], no intercept, weights 1/se^2 of the network shift; CI by resampling settings
    rows = [('P', k, R['pilot'][k]) for k in R['pilot']] + [('F', k, R['fresh'][k]) for k in R['fresh']]
    for cn in ('q', 'g'):
        for m in NETS:
            yv = np.array([r[cn][m]['d'] for _, _, r in rows]); se = np.array([r[cn][m]['se'] for _, _, r in rows])
            X2 = np.array([[r[cn]['efn_k']['d'], r[cn]['lr_n']['d']] for _, _, r in rows])
            X3 = np.array([[r[cn]['efn_k']['d'], r[cn]['lr_n']['d'], r[cn]['lr_ptd']['d']] for _, _, r in rows])
            res = {}
            for nm, X in (('efn_lrn', X2), ('efn_lrn_lrptd', X3)):
                w = 1 / np.maximum(se, 1e-3) ** 2

                def fit(ix):
                    sw = np.sqrt(w[ix]); return np.linalg.lstsq(X[ix] * sw[:, None], yv[ix] * sw, rcond=None)[0]
                c = fit(np.arange(len(yv))); r2 = np.random.default_rng(7)
                bs = np.array([fit(r2.integers(0, len(yv), len(yv))) for _ in range(2000)])
                pred = X @ c; ss_res = float((w * (yv - pred) ** 2).sum()); ss_tot = float((w * (yv - np.average(yv, weights=w)) ** 2).sum())
                res[nm] = dict(coef=c.tolist(), ci=[np.percentile(bs[:, j], [2.5, 97.5]).tolist() for j in range(X.shape[1])],
                               R2=1 - ss_res / ss_tot, chi2=ss_res, ndf=len(yv) - X.shape[1])
            # EFN-only model (H0) fit quality for comparison
            sw = np.sqrt(1 / np.maximum(se, 1e-3) ** 2); a1 = np.linalg.lstsq(X2[:, :1] * sw[:, None], yv * sw, rcond=None)[0]
            res['efn_only'] = dict(coef=a1.tolist(), chi2=float(((yv - X2[:, :1] @ a1) ** 2 / np.maximum(se, 1e-3) ** 2).sum()), ndf=len(yv) - 1)
            R['regression'].setdefault(cn, {})[m] = res
            log('regression', cn, m, {k: [round(x, 3) for x in v['coef']] for k, v in res.items()}, 'R2', round(res['efn_lrn']['R2'], 3))
    R['settings_rows'] = [f'{a_}|{k}' for a_, k, _ in rows]
    R['holm'] = dict(n_tests=len(R['tests']), n_pass_005=int(sum(t['holm'] < 0.05 for t in R['tests'])))
    (RES / 'h54_verify.json').write_text(json.dumps(R, indent=1, default=float)); log('wrote h54_verify.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', required=True, choices=['gen', 'score', 'stats'])
    ap.add_argument('--jets', type=int, default=8000); ap.add_argument('--cpu', type=int, default=8)
    ap.add_argument('--boot', type=int, default=1000)
    a = ap.parse_args()
    if a.step == 'gen': step_gen(a)
    elif a.step == 'score': step_score(a)
    else:
        Z = dict(np.load(RES / 'h54_verify_scores.npz'))
        sn = sorted({k[2:] for k in Z if k.startswith('y|')})
        stats(Z, sn, a.boot)
