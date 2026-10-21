"""H56 verify (loop D): fresh-sample rerun of the H56 axis-direction and count seed tests with a larger untrained (epoch-0) band.

Axis step (H39 protocol through interp.h39_axis_null, unchanged code paths): fresh samples Top Landscape test
data_top.sample(PER/class, seed 5691) and QG Pythia test data_qg.sample(PER/class, seed 5692). Networks: s0 and s1 (all
finished seed-1 trainings, incl. QG ParT_kin s1 final best checkpoint); untrained band with fresh seeds: weaver ParT
(NINIT_PART nets) and ParticleNet (NINIT nets) built after torch.manual_seed(2000 + k); EFN / PFN: energyflow default init
(he_uniform kernels U(+-sqrt(6/fan_in)), zero biases) drawn in numpy with rng 2000 + k, same tensor shapes as
models/h39_init/<arch>_init0 (checked against those stored inits). NR random directions. Shuffled-label nets omitted (constant).
Count step (H24v protocol through h24_qg_generator_models / h24_verify): fresh Pythia sample h14.load_sample('py', PERC, 5693),
H24.SEED = 5693 for the edits; ParT_kin / PN_kin s0, s1, Herwig-trained (qg_herwig/*_kin best), PFN / EFN s0, s1, init bands.
Run: ./sync.sh && ./part_job.sh d-h56-verify 'python -m interp.h56_verify'
Outputs: $AI_RESULTS/physics/h56_verify.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import h39_axis_null as H39   # noqa: E402

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
SMOKE = bool(os.environ.get('H56V_SMOKE'))
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
TAG = 'h56_verify_smoke' if SMOKE else 'h56_verify'
V = H39.V
PER = 100 if SMOKE else 3000
NR = 3 if SMOKE else 10
NINIT = 3 if SMOKE else 20
NINIT_PART = 2 if SMOKE else 12
PERC = 100 if SMOKE else 1000
B = 300 if SMOKE else 2000
H39.PER, H39.NR, H39.NINIT, H39.B, H39.TAG = PER, NR, NINIT, B, TAG
H39.H36.PER_MAIN, H39.H36.PER_CAL = PER, 0
H39.H36.SEEDS = dict(top=5691, py=5692, hw=5692)
STEPS = os.environ.get('H56V_STEPS', 'axis,count').split(',')
EFDIR = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics' / 'h56v_ef_init'


def make_ef_inits():
    """he_uniform / zero-bias EF inits with the shapes of the stored energyflow inits; returns check numbers"""
    chk = {}
    for a in ('pfn', 'efn'):
        W0 = np.load(V / 'models' / 'h39_init' / f'{a}_init0' / 'weights.npz')
        # check: stored init kernels are consistent with he_uniform (max |w| <= sqrt(6/fan_in), std ~ limit/sqrt(3)), biases 0
        rat = [float(np.abs(W0[k]).max() / np.sqrt(6 / W0[k].shape[0])) for k in W0.files if k.endswith('_0')]
        srat = [float(W0[k].std() / (np.sqrt(6 / W0[k].shape[0]) / np.sqrt(3))) for k in W0.files if k.endswith('_0')]
        bmax = max(float(np.abs(W0[k]).max()) for k in W0.files if k.endswith('_1'))
        chk[a] = dict(max_over_limit=rat, std_over_expected=srat, max_abs_bias=bmax)
        for k in range(NINIT):
            r = np.random.default_rng(2000 + k); out = {}
            for key in W0.files:
                sh = W0[key].shape
                if key.endswith('_0'): lim = np.sqrt(6 / sh[0]); out[key] = r.uniform(-lim, lim, sh).astype(np.float32)
                else: out[key] = np.zeros(sh, np.float32)
            d = EFDIR / f'{a}_init{k}'; d.mkdir(parents=True, exist_ok=True); np.savez(d / 'weights.npz', **out)
    log('EF init check', json.dumps(chk))
    return chk


def registry(ds, dev):
    import torch
    E = []; dsd = 'top' if ds == 'top' else 'qg'

    def add_w(name, arch, var, ckpt=None, seed=None):
        net = H39.build_weaver(arch, seed)
        if ckpt is not None: net.load_state_dict(H39.H4._strip(torch.load(ckpt, map_location='cpu')), strict=True)
        E.append(dict(name=name, arch=arch, var=var, kind='weaver', net=net.eval().to(dev), info=dict(checkpoint=str(ckpt), seed=seed)))

    def add_ef(name, arch, var, path):
        E.append(dict(name=name, arch=arch, var=var, kind='ef', net=H39.ef_from_npz(path, arch.lower()).eval().to(dev), info=dict(weights=str(path))))
    if ds == 'top':
        add_w('ParT_s0', 'ParT', 's0', H39.RES / 'h4_ckpt_part_kin.pt'); add_w('PN_s0', 'PN', 's0', H39.RES / 'h4_ckpt_pnet_kin.pt')
    else:
        add_w('ParT_s0', 'ParT', 's0', V / 'models/qg/part_kin/net_best_epoch_state.pt')
        add_w('PN_s0', 'PN', 's0', V / 'models/qg/pnet_kin/net_best_epoch_state.pt')
    for arch, wd in (('PN', 'pnet_kin'), ('ParT', 'part_kin')):
        d = V / 'models' / dsd / f'{wd}_s1'
        if d.exists():
            ok, ck = H39._weaver_done(d, 's1')
            if ok and ck.exists(): add_w(f'{arch}_s1', arch, 's1', ck)
            else: log(f'{ds}: {d.name} not finished, skipped')
    for a in ('PFN', 'EFN'):
        add_ef(f'{a}_s0', a, 's0', V / 'models' / dsd / a.lower() / 'weights.npz')
        p = V / 'models' / dsd / f'{a.lower()}_s1' / 'weights.npz'
        if p.exists(): add_ef(f'{a}_s1', a, 's1', p)
    for k in range(NINIT):
        if k < NINIT_PART: add_w(f'ParT_init{k}', 'ParT', 'init', seed=2000 + k)
        add_w(f'PN_init{k}', 'PN', 'init', seed=2000 + k)
        for a in ('PFN', 'EFN'): add_ef(f'{a}_init{k}', a, 'init', EFDIR / f'{a.lower()}_init{k}' / 'weights.npz')
    log(f'{ds}: {len(E)} networks')
    return E


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    return adj


def signflip(d, P=20000, seed=5611):
    r = np.random.default_rng(seed); obs = abs(d.sum()); cnt = 0
    for _ in range(P // 1000):
        sg = r.choice(np.array([-1.0, 1.0]), (1000, len(d))); cnt += int((np.abs(sg @ d) >= obs - 1e-9).sum())
    return (cnt + 1) / (P + 1)


def boot_ci(d, idx):
    s = np.concatenate([d[idx[c:c + 250]].mean(1) for c in range(0, len(idx), 250)])
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def step_axis(R, tests):
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    R['ef_init_check'] = make_ef_inits()
    H39.registry = registry
    SS = {'top': H39.H36.load_top(), 'py': H39.H36.load_qg('py')}
    R['axis'] = {}
    for ds in ('top', 'py'):
        res, pj = H39.run_ds(ds, SS[ds], dev)
        M = res['models']; J = len(SS[ds]['y']); idx = np.random.default_rng(5612).integers(0, J, (B, J)).astype(np.int32)
        out = dict(n=J, sample=res['sample'], checks=res['checks'],
                   accuracy={nm: dict(acc=m['acc'], auc=m['auc'], score_std=m['score_std']) for nm, m in M.items() if m['variant'] != 'init'},
                   init_acc_range={a: [min(m['acc'] for m in M.values() if m['arch'] == a and m['variant'] == 'init'),
                                       max(m['acc'] for m in M.values() if m['arch'] == a and m['variant'] == 'init')] for a in H39.ARCHS},
                   cells={})
        for arch in H39.ARCHS:
            for c in H39.CALLS:
                for a in H39.AX:
                    init = [m['calls'][c]['axes'][a]['D_pp'] for m in M.values() if m['arch'] == arch and m['variant'] == 'init'
                            and m['calls'][c]['axes'][a]['nondegenerate']]
                    cell = dict(init=dict(n=len(init), values=init, **(dict(min=min(init), max=max(init), mean=float(np.mean(init)),
                                                                              sd=float(np.std(init, ddof=1))) if len(init) > 1 else {})), nets={})
                    for nm in (f'{arch}_s0', f'{arch}_s1'):
                        if nm not in M: continue
                        v = M[nm]['calls'][c]['axes'][a]; nt = H39.null_test(v['D_pp'], init)
                        cell['nets'][nm] = dict(D_pp=v['D_pp'], D_ci=v['D_ci'], pct_axis=v['pct_axis'], pct_R=v['pct_R'], vs_init=nt)
                        if 'p_pred' in nt: tests.append((f'axis {ds} {arch} {c} {a} {nm} vs init', nt['p_pred']))
                    if f'{arch}_s0' in M and f'{arch}_s1' in M:
                        d = (pj[f'{ds}|d|{c}|{a}|{arch}_s1'].astype(float) - pj[f'{ds}|d|{c}|{a}|{arch}_s0'].astype(float)) * 100
                        p = signflip(d) if d.std() > 0 else 1.0
                        cell['s1_minus_s0'] = dict(diff_pp=float(d.mean()), ci=boot_ci(d, idx), p_signflip=p)
                        tests.append((f'axis {ds} {arch} {c} {a} s1-s0', p))
                    out['cells'][f'{arch}|{c}|{a}'] = cell
            e = out['cells'][f'{arch}|wp50|WTA']
            log(f'[{ds}] {arch} wp50 WTA ' + json.dumps({k: round(v['D_pp'], 2) for k, v in e['nets'].items()})
                + f" init n={e['init']['n']} [{e['init'].get('min', np.nan):+.2f},{e['init'].get('max', np.nan):+.2f}]")
        R['axis'][ds] = out
        (RES / f'{TAG}.json').write_text(json.dumps(R, indent=1, default=float))
        del res, pj
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def step_count(R, tests):
    import torch
    import interp.h24_qg_generator_models as H24
    from interp import h14_qg_generator as H14
    from interp import h5_qg_counting as H5
    from interp import h24_verify as H24V
    from scipy.stats import nbinom
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED = 5693; H24.SEED = SEED; KS = (1, 2, 3, 4, 5)
    if not EFDIR.exists(): make_ef_inits()
    L0, y, inf = H14.load_sample('py', PERC, SEED); J = len(y)
    E = H24.make_edits(L0, y, 'py')
    nmax = max(len(a) for a in E['sp5'][0]); log(f'count: {J} jets, max N after sp5 {nmax}')
    Mo = H5.Models(dev)

    def addw(nm, arch, ck=None, seed=None):
        if seed is not None: torch.manual_seed(seed)
        net = H24V.build_net(arch)
        if ck is not None: net.load_state_dict(Mo._sd(ck), strict=True)
        Mo.m[nm] = ('weaver', arch, False, net.eval().to(dev))
    q = V / 'models' / 'qg'
    addw('ParT_s0', 'part', q / 'part_kin/net_best_epoch_state.pt'); addw('PN_s0', 'pn', q / 'pnet_kin/net_best_epoch_state.pt')
    addw('ParT_s1', 'part', q / 'part_kin_s1/net_best_epoch_state.pt'); addw('PN_s1', 'pn', q / 'pnet_kin_s1/net_best_epoch_state.pt')
    addw('ParT_hw', 'part', V / 'models/qg_herwig/part_kin/net_best_epoch_state.pt'); addw('PN_hw', 'pn', V / 'models/qg_herwig/pnet_kin/net_best_epoch_state.pt')
    for a in ('PFN', 'EFN'):
        Mo.add_torch(f'{a}_s0', H39.ef_from_npz(q / a.lower() / 'weights.npz', a.lower()), {})
        Mo.add_torch(f'{a}_s1', H39.ef_from_npz(q / f'{a.lower()}_s1' / 'weights.npz', a.lower()), {})
    for k in range(NINIT):
        if k < NINIT_PART: addw(f'ParT_init{k}', 'part', seed=2000 + k)
        addw(f'PN_init{k}', 'pn', seed=2000 + k)
        for a in ('PFN', 'EFN'): Mo.add_torch(f'{a}_init{k}', H39.ef_from_npz(EFDIR / f'{a.lower()}_init{k}' / 'weights.npz', a.lower()), {})
    names = list(Mo.m)
    lrn = json.loads((RES / 'h24_base.json').read_text())['lrn']['py']
    T = nbinom.logpmf(np.arange(400), lrn['nb_g']['r'], lrn['nb_g']['p']) - nbinom.logpmf(np.arange(400), lrn['nb_q']['r'], lrn['nb_q']['p'])
    S = {}
    for cn in ['base'] + [f'sp{k}' for k in KS] + [f'ct{k}' for k in KS]:
        Le = L0 if cn == 'base' else E[cn][0]; Jt = H5.from_list(Le, Lpad=max(180, nmax))
        for nm in names: S[(cn, nm)] = Mo.logit(nm, Jt)
        S[(cn, 'LR_N')] = T[np.minimum([len(a) for a in Le], 399)].astype(float)
        log(f'count: {cn} scored')
    ok = E['sp1'][1].astype(bool); n = int(ok.sum())
    idx = np.random.default_rng(5613).integers(0, n, (B, n)).astype(np.int32)
    out = dict(sample=dict(per_class=PERC, seed=SEED, n=J, n_applied=n, idx_first5=inf['idx'][:5].tolist(), max_n_after_sp5=nmax), models={}, paired={})
    PJ = {}
    for nm in names + ['LR_N']:
        d0 = lambda c: S[(c, nm)] - S[('base', nm)]
        sp = H24V._slope(np.stack([d0(f'sp{k}') for k in KS], 1), KS); ct = H24V._slope(np.stack([d0(f'ct{k}') for k in KS], 1), KS)
        d = (sp - ct)[ok]; PJ[nm] = d
        e = dict(sp_minus_ct=float(d.mean()), ci=boot_ci(d, idx), sp_raw=float(sp[ok].mean()), ct_raw=float(ct[ok].mean()))
        if nm != 'LR_N': e['acc'] = float(np.mean((S[('base', nm)] < 0) == (y == 1)))
        out['models'][nm] = e
        if 'init' not in nm: log(f'count {nm:10s} acc {e.get("acc", np.nan):.4f} sp-ct {e["sp_minus_ct"]:+.5f} {np.round(e["ci"], 5).tolist()}')
    for arch in ('ParT', 'PN', 'PFN', 'EFN'):
        iv = [out['models'][f'{arch}_init{k}']['sp_minus_ct'] for k in range(NINIT) if f'{arch}_init{k}' in out['models']]
        out[f'init_band_{arch}'] = dict(n=len(iv), min=min(iv), max=max(iv), mean=float(np.mean(iv)), sd=float(np.std(iv, ddof=1)))
        for v in ('s0', 's1'):
            out['models'][f'{arch}_{v}']['vs_init'] = H39.null_test(out['models'][f'{arch}_{v}']['sp_minus_ct'], iv)
            tests.append((f'count {arch}_{v} vs init', out['models'][f'{arch}_{v}']['vs_init']['p_pred']))
        pairs = [(f'{arch}_s1', f'{arch}_s0')] + ([(f'{arch}_hw', f'{arch}_s0'), (f'{arch}_hw', f'{arch}_s1')] if arch in ('ParT', 'PN') else [])
        for b_, a_ in pairs:
            d = PJ[b_] - PJ[a_]; p = signflip(d) if d.std() > 0 else 1.0
            out['paired'][f'{b_} - {a_}'] = dict(diff=float(d.mean()), ci=boot_ci(d, idx), p_signflip=p)
            tests.append((f'count {b_} - {a_}', p))
    R['count'] = out


def main():
    RES.mkdir(parents=True, exist_ok=True)
    R = dict(node='H56 verify', smoke=SMOKE, per_class=PER, NR=NR, n_init=NINIT, n_init_part=NINIT_PART, seeds=dict(top=5691, py=5692, count=5693))
    tests = []
    if 'axis' in STEPS: step_axis(R, tests)
    if 'count' in STEPS: step_count(R, tests)
    adj = holm([t[1] for t in tests])
    R['holm'] = [dict(test=t[0], p=float(t[1]), holm=float(h)) for t, h in zip(tests, adj)]
    (RES / f'{TAG}.json').write_text(json.dumps(R, indent=1, default=float)); log(f'wrote {TAG}.json; Holm over {len(tests)}')


if __name__ == '__main__':
    main()
