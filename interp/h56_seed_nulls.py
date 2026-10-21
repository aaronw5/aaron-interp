"""H56 (loop D; claims D4m, D4 direction part, D16, D7): seed and shuffled-label nulls for the axis-direction (H39), charge
relabel (H42) and count (H24v / H26) results.

Networks compared per architecture (all trained on the same data with the authors' settings unless stated):
  s0     the trainings used so far (H2 / H4 / H36 / H39 models)
  s1     second training seed (torch seed 1 before the model is built; energyflow seed 1)
  shuf   labels independent of the class (weaver: label = hash(jet pT, eta, phi) mod 2; energyflow: permuted labels), same steps
  s1p / shufp  provisional: the newest finished epoch checkpoint of a training that is still running (epoch k of 20), with
         the s0 checkpoint of the same epoch k (s0e) as its epoch-matched seed-0 partner
  init   epoch-0 networks: 6 per architecture (weaver ParT / ParticleNet built after torch.manual_seed(1000 + k); EFN / PFN
         built by energyflow with seed k, models/h39_init/)
(1) axis direction (H39 protocol, re-using interp.h39_axis_null): Top Landscape test (data_top.sample(5000/class, seed 3901)),
    QG Pythia test (data_qg.sample(5000/class, seed 3911)). Calls at sign(log-odds), eps_S = 0.3, 0.5. Per network x axis
    (WTA, hardest particle HP) x call: D = % calls changed about the axis - % changed by a translation of equal per-jet length
    in a uniformly random direction (mean of 20 directions). Tests: s0, s1, shuf vs the init band (z, t prediction p); seed
    difference D(s1) - D(s0) with a paired jet bootstrap CI; McNemar on the paired per-jet indicators "call changed about the
    axis" (s1 vs s0, shuf vs s0).
(2) charge relabel (H42 protocol): needs the kinpid s1 / shuf trainings; none has finished (see report) -> not run here.
(3) count response (H24v protocol): Pythia jets, h14.load_sample('py', 2000/class, seed 7243), H24 edits with SEED = 7243:
    sp_k = exact collinear split of k (1..5) soft particles into two halves of the same direction (energy flow unchanged,
    N + k), ct_k = Delta N = 0 control at matched pT^D. Per jet: slope over k of Delta logit(sp_k) - slope of Delta logit(ct_k)
    (log-odds toward gluon per added particle). Models: ParT_kin / PN_kin s0, s1, shuf (+ provisional), EFN / PFN s0, s1,
    shuf, 6 init each, part_kin weight-permuted (H24v null), LR(N) (negative-binomial likelihood ratio in N, h24_base.json).
Statistics: 95 % bootstrap over jets (B = 2000), sign-flip permutation p for mean 0, seed spread |s0 - s1| with paired
bootstrap CI and sign-flip p, Holm over all tests of the run.
Run: ./sync.sh && ./part_job.sh d-h56-run 'python -m interp.h56_seed_nulls'      (H56_SMOKE=1: small samples)
Outputs: $AI_RESULTS/physics/h56_seed_nulls.json, h56_axis_null.json (H39-format), h56_perjet.npz, h56_seed_nulls.png
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
SMOKE = bool(os.environ.get('H56_SMOKE'))
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
TAG = 'h56_smoke' if SMOKE else 'h56'
V = H39.V
B = 300 if SMOKE else 2000
if SMOKE:
    H39.PER, H39.NR, H39.NINIT, H39.B = 100, 3, 2, 300
    H39.H36.PER_MAIN = 100
H39.TAG = TAG
NINIT = H39.NINIT
STEPS = os.environ.get('H56_STEPS', 'axis,count').split(',')


# ============================================================================================================== registry
def _latest_epoch(d, min_age=180):
    eps = []
    for f in d.glob('net_epoch-*_state.pt'):
        if time.time() - f.stat().st_mtime > min_age: eps.append(int(f.name.split('-')[1].split('_')[0]))
    return max(eps) if eps else None


def weaver_variants(dsd):
    """-> list of (name, arch, var, ckpt, info) for the trained weaver kin networks of one dataset"""
    out = []
    s0 = dict(ParT=V / 'models' / dsd / 'part_kin', PN=V / 'models' / dsd / 'pnet_kin')
    for arch, wd in (('PN', 'pnet_kin'), ('ParT', 'part_kin')):
        for var in ('s1', 'shuf'):
            d = V / 'models' / dsd / f'{wd}_{var}'
            if not d.exists(): continue
            ok, ck = H39._weaver_done(d, var)
            if ok and ck.exists():
                out.append((f'{arch}_{var}', arch, var, ck, dict(checkpoint=str(ck), status='final')))
                continue
            k = _latest_epoch(d)
            if k is None: log(f'{dsd}: {d.name}: no finished epoch'); continue
            ck = d / f'net_epoch-{k}_state.pt'
            out.append((f'{arch}_{var}p', arch, f'{var}p', ck, dict(checkpoint=str(ck), status=f'provisional, epoch {k} of 0-19', epoch=k)))
            if var == 's1' and dsd == 'qg':
                c0 = s0[arch] / f'net_epoch-{k}_state.pt'
                if c0.exists(): out.append((f'{arch}_s0e', arch, 's0e', c0, dict(checkpoint=str(c0), status=f'seed 0 at epoch {k} (partner of s1p)', epoch=k)))
            log(f'{dsd}: {d.name} running: provisional epoch {k}')
    return out


def registry(ds, dev):
    """H39.registry with s1 / shuf / provisional variants; shuf named 'shufX' inside H39 so that its null is init only"""
    import torch
    E = []; dsd = 'top' if ds == 'top' else 'qg'

    def add_w(name, arch, var, ckpt=None, seed=None, info=None):
        net = H39.build_weaver(arch, seed)
        if ckpt is not None: net.load_state_dict(H39.H4._strip(torch.load(ckpt, map_location='cpu')), strict=True)
        E.append(dict(name=name, arch=arch, var=var, kind='weaver', net=net.eval().to(dev), info=info or dict(checkpoint=str(ckpt), seed=seed)))

    def add_ef(name, arch, var, path, info=None):
        E.append(dict(name=name, arch=arch, var=var, kind='ef', net=H39.ef_from_npz(path, arch.lower()).eval().to(dev), info=info or dict(weights=str(path))))
    if ds == 'top':
        add_w('ParT_s0', 'ParT', 's0', H39.RES / 'h4_ckpt_part_kin.pt'); add_w('PN_s0', 'PN', 's0', H39.RES / 'h4_ckpt_pnet_kin.pt')
    else:
        add_w('ParT_s0', 'ParT', 's0', V / 'models/qg/part_kin/net_best_epoch_state.pt')
        add_w('PN_s0', 'PN', 's0', V / 'models/qg/pnet_kin/net_best_epoch_state.pt')
    for a in ('PFN', 'EFN'):
        d = V / 'models' / dsd / a.lower(); add_ef(f'{a}_s0', a, 's0', d / 'weights.npz', json.loads((d / 'info.json').read_text()))
    for name, arch, var, ck, info in weaver_variants(dsd):
        add_w(name, arch, 'shufX' if var == 'shuf' else var, ck, info=info)
    for a in ('PFN', 'EFN'):
        for var in ('s1', 'shuf'):
            d = V / 'models' / dsd / f'{a.lower()}_{var}'
            if (d / 'info.json').exists():
                add_ef(f'{a}_{var}', a, 'shufX' if var == 'shuf' else var, d / 'weights.npz', json.loads((d / 'info.json').read_text()))
    for k in range(NINIT):
        for arch in ('ParT', 'PN'): add_w(f'{arch}_init{k}', arch, 'init', seed=1000 + k)
        for a in ('PFN', 'EFN'):
            p = V / 'models' / 'h39_init' / f'{a.lower()}_init{k}' / 'weights.npz'
            if p.exists(): add_ef(f'{a}_init{k}', a, 'init', p)
    log(f'{ds}: {len(E)} networks: ' + ', '.join(f'{e["name"]}({e["var"]})' for e in E))
    return E


# ============================================================================================================== stats
def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    return adj


def boot_ci(d, idx):
    s = np.concatenate([d[idx[c:c + 250]].mean(1) for c in range(0, len(idx), 250)])
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def signflip(d, P=20000, seed=5601):
    r = np.random.default_rng(seed); obs = abs(d.sum()); cnt = 0
    for _ in range(P // 1000):
        sg = r.choice(np.array([-1.0, 1.0]), (1000, len(d))); cnt += int((np.abs(sg @ d) >= obs - 1e-9).sum())
    return (cnt + 1) / (P + 1)


def mcnemar(a, b):
    from scipy.stats import binomtest
    n01 = int((~a & b).sum()); n10 = int((a & ~b).sum())
    return dict(n_only_b=n01, n_only_a=n10, p=float(binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0)


# ============================================================================================================== (1) axis
def step_axis(R):
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    H39.registry = registry
    SS = {'top': H39.H36.load_top(), 'py': H39.H36.load_qg('py')}
    A = dict(datasets={}); tests = []
    for ds in ('top', 'py'):
        res, pj = H39.run_ds(ds, SS[ds], dev)
        y = SS[ds]['y']; J = len(y); idx = np.random.default_rng(5602).integers(0, J, (B, J)).astype(np.int32)
        M = res['models']; sgn = (lambda l: l) if ds == 'top' else (lambda l: -l)
        # per-jet call-change indicators about each axis (same calls as H39.analyse)
        chg = {}
        for nm, m in M.items():
            l0 = pj[f'{ds}|E|{nm}'].astype(float); thr = m['wp_threshold']
            call = {'sign': lambda l: l > 0, **{c: (lambda l, t=thr[c]: sgn(l) >= t) for c in H39.WP}}
            for c in H39.CALLS:
                for a in H39.AX: chg[(nm, c, a)] = call[c](pj[f'{ds}|{a}|{nm}'].astype(float)) != call[c](l0)
        out = dict(n=J, accuracy={nm: dict(acc=m['acc'], auc=m['auc'], variant=m['variant'], info=m['info'], score_std=m['score_std']) for nm, m in M.items()},
                   cells={})
        for arch in H39.ARCHS:
            for c in H39.CALLS:
                for a in H39.AX:
                    init = {nm: m['calls'][c]['axes'][a]['D_pp'] for nm, m in M.items() if m['arch'] == arch and m['variant'] == 'init'
                            and m['calls'][c]['axes'][a]['nondegenerate']}
                    cell = dict(init_band=dict(values=init, **({} if len(init) < 2 else dict(min=min(init.values()), max=max(init.values()),
                                                                                                 mean=float(np.mean(list(init.values()))),
                                                                                                 sd=float(np.std(list(init.values()), ddof=1))))),
                                nets={})
                    for nm, m in M.items():
                        if m['arch'] != arch or m['variant'] == 'init': continue
                        v = m['calls'][c]['axes'][a]
                        e = dict(variant=m['variant'], D_pp=v['D_pp'], D_ci=v['D_ci'], pct_axis=v['pct_axis'], pct_R=v['pct_R'],
                                 nondegenerate=v['nondegenerate'], vs_init=H39.null_test(v['D_pp'], list(init.values())))
                        cell['nets'][nm] = e
                        if 'p_pred' in e['vs_init'] and m['variant'] in ('s0', 's1', 'shufX', 's1p', 'shufp'):
                            tests.append(('axis', ds, arch, c, a, nm, 'vs_init', e['vs_init']['p_pred']))
                    # seed difference and shuffled - s0, paired over jets
                    pairs = [(f'{arch}_s1', f'{arch}_s0'), (f'{arch}_s1p', f'{arch}_s0e'), (f'{arch}_shuf', f'{arch}_s0'), (f'{arch}_shufp', f'{arch}_s0')]
                    cell['paired'] = {}
                    for b_, a_ in pairs:
                        if b_ not in M or a_ not in M: continue
                        d = (pj[f'{ds}|d|{c}|{a}|{b_}'].astype(float) - pj[f'{ds}|d|{c}|{a}|{a_}'].astype(float)) * 100
                        mc = mcnemar(chg[(a_, c, a)], chg[(b_, c, a)])
                        p = signflip(d) if d.std() > 0 else 1.0
                        cell['paired'][f'{b_} - {a_}'] = dict(diff_pp=float(d.mean()), ci=boot_ci(d, idx), abs_diff=float(abs(d.mean())), p_signflip=p,
                                                              mcnemar_axis_change=mc, same_sign=bool(np.sign(M[b_]['calls'][c]['axes'][a]['D_pp']) == np.sign(M[a_]['calls'][c]['axes'][a]['D_pp'])))
                        tests.append(('axis', ds, arch, c, a, f'{b_} - {a_}', 'paired', p))
                        tests.append(('axis', ds, arch, c, a, f'{b_} - {a_}', 'mcnemar', mc['p']))
                    out['cells'][f'{arch}|{c}|{a}'] = cell
            log(f'[{ds}] {arch} wp50 WTA: ' + json.dumps({k: round(v['D_pp'], 2) for k, v in out['cells'][f"{arch}|wp50|WTA"]['nets'].items()})
                + f' init {sorted(np.round(list(out["cells"][f"{arch}|wp50|WTA"]["init_band"]["values"].values()), 2))}')
        out['checks'] = res['checks']; out['lorentznet'] = res.get('lorentznet'); out['sample'] = res['sample']
        A['datasets'][ds] = out
        np.savez_compressed(RES / f'{TAG}_axis_perjet_{ds}.npz', **{k: v for k, v in pj.items()})
        R['axis'] = A; (RES / f'{TAG}_seed_nulls.json').write_text(json.dumps(R, indent=1, default=float))
        del res, pj
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return tests


# ============================================================================================================== (3) count
def step_count(R):
    import torch
    import interp.h24_qg_generator_models as H24
    from interp import h14_qg_generator as H14
    from interp import h5_qg_counting as H5
    from interp import h24_verify as H24V
    from scipy.stats import nbinom
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED = 7243; H24.SEED = SEED; per = 200 if SMOKE else 2000; KS = (1, 2, 3, 4, 5)
    L0, y, inf = H14.load_sample('py', per, SEED); J = len(y)
    E = H24.make_edits(L0, y, 'py')
    log(f'count: {J} Pythia jets (seed {SEED}); idx[:5] {inf["idx"][:5].tolist()}; applied sp1 {int(E["sp1"][1].sum())}')
    Mo = H5.Models(dev); nets = {}; info = {}

    def addw(nm, arch, ck=None, seed=None, inf_=None):
        if seed is not None: torch.manual_seed(seed)
        net = H24V.build_net(arch)
        if ck is not None: net.load_state_dict(Mo._sd(ck), strict=True)
        Mo.m[nm] = ('weaver', arch, False, net.eval().to(dev)); info[nm] = inf_ or dict(checkpoint=str(ck), seed=seed)
    addw('ParT_s0', 'part', V / 'models/qg/part_kin/net_best_epoch_state.pt'); addw('PN_s0', 'pn', V / 'models/qg/pnet_kin/net_best_epoch_state.pt')
    for name, arch, var, ck, inf_ in weaver_variants('qg'):
        addw(name, 'part' if arch == 'ParT' else 'pn', ck, inf_=dict(variant=var, **inf_))
    for a in ('PFN', 'EFN'):
        for var in ('s0', 's1', 'shuf'):
            d = V / 'models' / 'qg' / (a.lower() if var == 's0' else f'{a.lower()}_{var}')
            if (d / 'info.json').exists():
                Mo.add_torch(f'{a}_{var}', H39.ef_from_npz(d / 'weights.npz', a.lower()), dict(weights=str(d)))
                info[f'{a}_{var}'] = json.loads((d / 'info.json').read_text())
    for k in range(NINIT):
        addw(f'ParT_init{k}', 'part', seed=1000 + k); addw(f'PN_init{k}', 'pn', seed=1000 + k)
        for a in ('PFN', 'EFN'):
            p = V / 'models' / 'h39_init' / f'{a.lower()}_init{k}' / 'weights.npz'
            if p.exists(): Mo.add_torch(f'{a}_init{k}', H39.ef_from_npz(p, a.lower()), dict(weights=str(p))); info[f'{a}_init{k}'] = str(p)
    Mo.add_permuted('ParT_perm', 'ParT_s0'); info['ParT_perm'] = 'ParT_s0 with permuted weights (seed 0), H24v null'
    names = list(Mo.m)
    lrn = json.loads((RES / 'h24_base.json').read_text())['lrn']['py']
    T = nbinom.logpmf(np.arange(400), lrn['nb_g']['r'], lrn['nb_g']['p']) - nbinom.logpmf(np.arange(400), lrn['nb_q']['r'], lrn['nb_q']['p'])
    S = {}
    for cn in ['base'] + [f'sp{k}' for k in KS] + [f'ct{k}' for k in KS]:
        Le = L0 if cn == 'base' else E[cn][0]; Jt = H5.from_list(Le, Lpad=180)
        for nm in names: S[(cn, nm)] = Mo.logit(nm, Jt)
        S[(cn, 'LR_N')] = T[np.minimum([len(a) for a in Le], 399)].astype(float)
        log(f'count: {cn} scored')
    ok = E['sp1'][1].astype(bool)
    idx = np.random.default_rng(5603).integers(0, int(ok.sum()), (B, int(ok.sum()))).astype(np.int32)
    out = dict(sample=dict(per_class=per, seed=SEED, n=J, n_applied=int(ok.sum()), idx_first5=inf['idx'][:5].tolist()), models={}, paired={}, info=info)
    tests = []; PJ = {}
    for nm in names + ['LR_N']:
        d0 = lambda c: S[(c, nm)] - S[('base', nm)]
        sp = H24V._slope(np.stack([d0(f'sp{k}') for k in KS], 1), KS); ct = H24V._slope(np.stack([d0(f'ct{k}') for k in KS], 1), KS)
        d = (sp - ct)[ok]; PJ[nm] = d
        e = dict(sp_minus_ct=float(d.mean()), ci=boot_ci(d, idx), sp_raw=float(sp[ok].mean()), ct_raw=float(ct[ok].mean()),
                 p_signflip=signflip(d) if d.std() > 0 else 1.0, score_std=float(S[('base', nm)].std()))
        if nm != 'LR_N': e.update(H5.roc_metrics(S[('base', nm)], y))
        out['models'][nm] = e
        if not nm.split('_')[1].startswith('init') and nm not in ('LR_N',):
            tests.append(('count', 'py', nm, 'sp-ct', '', nm, 'signflip', e['p_signflip']))
        log(f'count {nm:12s} acc {e.get("acc", np.nan):.4f} AUC {e.get("AUC", np.nan):.4f} sp-ct {e["sp_minus_ct"]:+.5f} {np.round(e["ci"], 5).tolist()} p {e["p_signflip"]:.2e}')
    for arch in ('ParT', 'PN', 'PFN', 'EFN'):
        iv = [out['models'][f'{arch}_init{k}']['sp_minus_ct'] for k in range(NINIT) if f'{arch}_init{k}' in out['models']]
        out[f'init_band_{arch}'] = dict(values=iv, min=min(iv), max=max(iv), mean=float(np.mean(iv)), sd=float(np.std(iv, ddof=1)) if len(iv) > 1 else None) if iv else None
        for b_, a_ in ((f'{arch}_s1', f'{arch}_s0'), (f'{arch}_s1p', f'{arch}_s0e'), (f'{arch}_shuf', f'{arch}_s0'), (f'{arch}_shufp', f'{arch}_s0')):
            if b_ in PJ and a_ in PJ:
                d = PJ[b_] - PJ[a_]; p = signflip(d) if d.std() > 0 else 1.0
                out['paired'][f'{b_} - {a_}'] = dict(diff=float(d.mean()), abs_diff=float(abs(d.mean())), ci=boot_ci(d, idx), p_signflip=p)
                tests.append(('count', 'py', arch, 'sp-ct', '', f'{b_} - {a_}', 'paired', p))
        for nm in [n for n in out['models'] if n.startswith(arch + '_') and 'init' not in n]:
            if iv and len(iv) > 1: out['models'][nm]['vs_init'] = H39.null_test(out['models'][nm]['sp_minus_ct'], iv)
    R['count'] = out
    np.savez_compressed(RES / f'{TAG}_count_perjet.npz', y=y[ok], **{k: v.astype(np.float32) for k, v in PJ.items()})
    return tests


# ============================================================================================================== figure
def fig(R):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    col = dict(s0='C3', s0e='#f4a3a3', s1='C0', s1p='#8cc4e8', shufX='C2', shufXp='#9bd49b', shuf='C2', shufp='#9bd49b')
    lab = dict(s0='seed 0', s0e='seed 0, same epoch as s1 (prov.)', s1='seed 1', s1p='seed 1 (prov., still training)',
               shufX='shuffled labels', shufXp='shuffled labels (prov.)', shuf='shuffled labels', shufp='shuffled labels (prov.)')
    f, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    seen = set()
    for j, ds in enumerate(('top', 'py')):
        ax = axs[j]; D = R.get('axis', {}).get('datasets', {}).get(ds)
        if D is None: continue
        for xi, arch in enumerate(H39.ARCHS):
            cell = D['cells'][f'{arch}|wp50|WTA']; iv = list(cell['init_band']['values'].values())
            if iv: ax.add_patch(plt.Rectangle((xi - 0.3, min(iv)), 0.6, max(iv) - min(iv) + 1e-9, color='0.88', zorder=0))
            ax.plot([xi - 0.25] * len(iv), iv, 'o', mfc='none', mec='0.5', ms=4)
            nets = list(cell['nets'].items())
            for q, (nm, e) in enumerate(nets):
                x = xi - 0.12 + 0.3 * q / max(1, len(nets) - 1) if len(nets) > 1 else xi
                lb = lab.get(e['variant']) if e['variant'] not in seen else None; seen.add(e['variant'])
                ax.errorbar(x, e['D_pp'], yerr=[[e['D_pp'] - e['D_ci'][0]], [e['D_ci'][1] - e['D_pp']]], fmt='o', color=col.get(e['variant'], 'k'), ms=5, capsize=2, label=lb)
        ax.axhline(0, color='0.5', lw=0.7); ax.set_xticks(range(4)); ax.set_xticklabels(['ParT', 'ParticleNet', 'PFN', 'EFN'])
        ax.set_title(('Top Landscape' if ds == 'top' else 'Quark–gluon (Pythia)') + ': WTA axis, calls at 50 % signal eff.', fontsize=9, loc='left')
        if j == 0: ax.set_ylabel('direction part D (percentage points of calls)\naxis shift − random-direction shift', fontsize=9)
    for a_ in axs[:2]: a_.legend(fontsize=7, loc='lower left')
    ax = axs[2]; C = R.get('count')
    if C:
        for xi, arch in enumerate(('ParT', 'PN', 'PFN', 'EFN')):
            b = C.get(f'init_band_{arch}')
            if b: ax.add_patch(plt.Rectangle((xi - 0.3, b['min']), 0.6, b['max'] - b['min'] + 1e-9, color='0.88', zorder=0)); ax.plot([xi - 0.25] * len(b['values']), b['values'], 'o', mfc='none', mec='0.5', ms=4)
            nets = [(n, e) for n, e in C['models'].items() if n.startswith(arch + '_') and 'init' not in n and 'perm' not in n]
            for q, (nm, e) in enumerate(nets):
                v = nm.split('_')[1]; v = {'shuf': 'shuf', 'shufp': 'shufp'}.get(v, v)
                x = xi - 0.12 + 0.3 * q / max(1, len(nets) - 1) if len(nets) > 1 else xi
                ax.errorbar(x, e['sp_minus_ct'], yerr=[[e['sp_minus_ct'] - e['ci'][0]], [e['ci'][1] - e['sp_minus_ct']]], fmt='o', color=col.get(v, 'k'), ms=5, capsize=2)
        lr = C['models']['LR_N']['sp_minus_ct']; ax.axhline(lr, color='C1', ls='--', lw=1); ax.text(3.4, lr, f'LR(N) {lr:+.3f}', color='C1', fontsize=7.5, ha='right', va='bottom')
        ax.axhline(0, color='0.5', lw=0.7); ax.set_xticks(range(4)); ax.set_xticklabels(['ParT_kin', 'PN_kin', 'PFN', 'EFN'])
        ax.set_title('Quark–gluon (Pythia): exact collinear split − ΔN = 0 control', fontsize=9, loc='left')
        ax.set_yscale('symlog', linthresh=0.01); ax.set_ylabel('Δ log-odds toward gluon per added particle\n(symmetric log scale, linear within ±0.01)', fontsize=9)
    f.suptitle('H56: trained networks (seed 0, seed 1, shuffled labels) against epoch-0 networks of the same architecture (grey band, open circles)', fontsize=10)
    f.tight_layout(); f.savefig(RES / f'{TAG}_seed_nulls.png', dpi=130); log(f'wrote {TAG}_seed_nulls.png')


def main():
    RES.mkdir(parents=True, exist_ok=True)
    R = dict(node='H56', smoke=SMOKE, steps=STEPS, charge=dict(status='not run: no kinpid s1 / shuf checkpoint finished; see report'))
    tests = []
    if 'axis' in STEPS: tests += step_axis(R)
    if 'count' in STEPS: tests += step_count(R)
    adj = holm([t[-1] for t in tests])
    R['holm'] = dict(n_tests=len(tests), tests=[dict(analysis=t[0], ds=t[1], arch=t[2], call=t[3], axis=t[4], net=t[5], kind=t[6], p=float(t[7]), holm=float(h))
                                                for t, h in zip(tests, adj)])
    (RES / f'{TAG}_seed_nulls.json').write_text(json.dumps(R, indent=1, default=float)); log(f'wrote {TAG}_seed_nulls.json, Holm over {len(tests)}')
    try: fig(R)
    except Exception as ex: log('figure failed', repr(ex))


if __name__ == '__main__':
    if '--fig' in sys.argv: fig(json.loads((RES / f'{TAG}_seed_nulls.json').read_text()))
    else: main()
