"""H40 verification (loop D, verifier). Re-runs the key H40 edits on a fresh Pythia KMT test sample (rows disjoint from the
H40 sample data_qg.sample(10000, 4001)), and adds a null ensemble the H40 run lacked: K weight-permuted copies of ParT kinpid
and of ParticleNet kinpid (seeds 1..K), to give the chance distribution of the direction and size of each contrast.

Sample  10,000 quark + 10,000 gluon random rows of the test split not in the H40 sample, seed 4777.
Edits   interp.h40_qg_pid_charge.make_edits(full=False): E1, E2k1, C2, E3g_4, E3n_4, C3_4, E3r_4, C3r_4, E4a/b/n, C4.
Models  part_kinpid, pnet_kinpid, lorentznet; part_perm_s1..sK, pnet_perm_s1..sK; references re-trained (seed 4777).
Extra contrasts  E2k1_signlead = (E2k1 - C2) x sign(charge of the hardest charged particle).
Output  $AI_RESULTS/physics/h40_verify.json, h40_verify_perjet.npz
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import h40_qg_pid_charge as H40  # noqa: E402

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
V = Path('/j-jepa-vol/aaron-interp')
RES = Path(os.environ.get('AI_RESULTS', V / 'results')) / 'physics'
SEEDV = 4777
K = int(os.environ.get('H40V_K', 6))
PER = int(os.environ.get('H40V_PER', 10000))


def fresh_sample():
    from interp import data_qg
    from interp import h5_qg_counting as H
    old = data_qg.sample(10000, 4001)['idx']
    d = data_qg.DST / 'test'; y = np.load(d / 'label.npy'); rng = np.random.default_rng(SEEDV)
    free = np.ones(len(y), bool); free[old] = False
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero((y == c) & free), PER, replace=False) for c in (0, 1)]))
    assert not np.isin(idx, old).any()
    S = dict(idx=idx, label=y[idx], full=np.asarray(np.load(d / 'full.npy', mmap_mode='r')[idx]),
             pid=np.asarray(np.load(d / 'pid.npy', mmap_mode='r')[idx]))
    return H.to_list(H.from_test(S)), y[idx].astype(int), idx


def add_perm(M, name, src, seed):
    import copy
    t = M.t; kind, arch, pidfeat, net = M.m[src]; g = t.Generator().manual_seed(seed)
    net2 = copy.deepcopy(net).cpu()
    with t.no_grad():
        for p in net2.parameters():
            flat = p.data.view(-1); flat.copy_(flat[t.randperm(flat.numel(), generator=g)])
    M.m[name] = (kind, arch, pidfeat, net2.eval().to(M.dev)); M.info[name] = dict(note=f'{src} permuted, seed {seed}')


def main():
    import torch
    from interp import h5_qg_counting as H
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; RES.mkdir(parents=True, exist_ok=True)
    log(f'device {dev}; per class {PER}; K {K}')
    R, rinfo = H40.train_refs(n_per_class=50000, seed=SEEDV)
    M = H.Models(dev)
    for name, arch in (('part_kinpid', 'part'), ('pnet_kinpid', 'pn')): M.add_weaver(name, arch, True)
    M.add_lnet()
    perms = []
    for s in range(1, K + 1):
        for src, tag in (('part_kinpid', 'part'), ('pnet_kinpid', 'pnet')):
            nm = f'{tag}_perm_s{s}'; add_perm(M, nm, src, s); perms.append(nm)
    nets = ['part_kinpid', 'pnet_kinpid', 'lorentznet'] + perms
    H40.NETS = nets; H40.TESTED = nets; H40.ZEROS = []
    L0, y, idx = fresh_sample(); J = len(y)
    log(f'fresh sample {J} jets, quark frac {y.mean():.3f}')
    E = H40.make_edits(L0, 'py_verify', full=False)
    log('applied ' + ', '.join(f'{k} {int(v[1].sum())}' for k, v in E.items()))
    out = dict(y=y, idx=idx)
    J0 = H.from_list(L0, Lpad=180); F0, shp = H40.all_feats(J0)
    for k, v in F0.items(): out[f'F|base|{k}'] = v.astype(np.float32)
    out['F|base|qlead'] = np.array([(lambda q: q[q != 0][0] if (q != 0).any() else 0.0)(H40.charge(a[:, 3])) for a in L0])
    for nm in nets: out[f'base|{nm}'] = -M.logit(nm, J0)
    for nm, v in H40.ref_scores(R, F0).items(): out[f'base|{nm}'] = v
    unedited = {nm: H40.roc(out[f'base|{nm}'], y) for nm in nets + H40.REFS}
    for nm in nets + H40.REFS: log(f'unedited {nm}: ' + ', '.join(f'{k} {v:.4f}' for k, v in unedited[nm].items()))
    for en, (Le, ok, nr) in E.items():
        Je = H.from_list(Le, Lpad=180); Fe, _ = H40.all_feats(Je, shp)
        assert np.allclose(Je['pt'], J0['pt']) and np.allclose(Je['eta'], J0['eta']) and np.allclose(Je['phi'], J0['phi'])
        out[f'ok|{en}'] = ok; out[f'nr|{en}'] = nr
        for k in ('Q03', 'Q05', 'fch', 'n_ch', 'n_K', 'n_p', 'n_gam', 'n_K0L', 'n_n', 'n_pi'): out[f'F|{en}|{k}'] = Fe[k].astype(np.float32)
        for nm in nets: out[f'{en}|{nm}'] = -M.logit(nm, Je)
        for nm, v in H40.ref_scores(R, Fe).items(): out[f'{en}|{nm}'] = v
        log(f'{en}: ' + '; '.join(f'{nm} {np.mean((out[f"{en}|{nm}"] - out[f"base|{nm}"])[ok]) if ok.any() else 0:+.4f}'
                                 for nm in ['part_kinpid', 'pnet_kinpid', 'lorentznet', 'part_perm_s1', 'LR_Q03', 'GB_sp']))
    S = H40.statistics(out, y, unedited, 'py', False, 1000)
    # signed E2k1 contrast
    bt = H40.Boot(B=1000); sl = np.sign(out['F|base|qlead'].astype(float)); ok = out['ok|E2k1'] & out['ok|C2'] & (sl != 0)
    S['E2k1_signlead'] = {}
    for nm in nets + H40.REFS:
        d = ((out[f'E2k1|{nm}'] - out[f'base|{nm}']) - (out[f'C2|{nm}'] - out[f'base|{nm}']))[ok] * sl[ok]
        S['E2k1_signlead'][nm] = dict(n=int(ok.sum()), mean=float(d.mean()), ci=bt.ci(d), p=bt.p(d),
                                      mean_over_sd=float(d.mean() / np.std(out[f'base|{nm}'])))
    # null ensembles
    keys = ['E1_signQ', 'E3g_4-C3', 'E3n_4-C3', 'E3r_4-C3r', 'gam2K0L_4']
    NULL = {}
    for tag, tr in (('part', 'part_kinpid'), ('pnet', 'pnet_kinpid')):
        ps = [p for p in perms if p.startswith(tag)]; z = {}
        for cn in keys + ['E2k1_signlead']:
            g = (lambda nm: S['E2k1_signlead'][nm]) if cn == 'E2k1_signlead' else (lambda nm: S['contrasts'][cn][nm])
            t = g(tr); v = [g(p)['mean'] for p in ps]; vs = [g(p)['mean_over_sd'] for p in ps]
            z[cn] = dict(trained=t['mean'], trained_over_sd=t['mean_over_sd'], perm_means=v, perm_over_sd=vs,
                         perm_max_abs=float(np.max(np.abs(v))), perm_max_abs_over_sd=float(np.max(np.abs(vs))),
                         frac_perm_same_sign=float(np.mean(np.sign(v) == np.sign(t['mean']))),
                         trained_exceeds_all_perm_logodds=bool(abs(t['mean']) > np.max(np.abs(v))),
                         trained_exceeds_all_perm_sd=bool(abs(t['mean_over_sd']) > np.max(np.abs(vs))))
        z['perm_auc'] = [unedited[p]['AUC'] for p in ps]; z['perm_score_sd'] = [float(np.std(out[f'base|{p}'])) for p in ps]
        z['trained_score_sd'] = float(np.std(out[f'base|{tr}']))
        NULL[tag] = z
        log(tag, json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if not isinstance(vv, list)}) for k, v in z.items()}, default=float))
    res = dict(sample=dict(J=J, per_class=PER, seed=SEEDV, disjoint_from='data_qg.sample(10000, 4001)'), refs=rinfo,
               models=M.info, unedited=unedited, stats=S, null=NULL, applied={k: int(v[1].sum()) for k, v in E.items()})
    (RES / 'h40_verify.json').write_text(json.dumps(res, indent=1, default=float))
    np.savez_compressed(RES / 'h40_verify_perjet.npz', **{k: (v.astype(np.float32) if v.dtype == np.float64 else v) for k, v in out.items()})
    log('wrote', RES / 'h40_verify.json')


if __name__ == '__main__':
    main()
