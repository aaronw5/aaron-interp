"""E25x verification (fresh sample; matched-budget fully random baselines at a; edit-specificity controls at a and S).
   python -m interp.e25x_verify --model {3f,16f} --part trained                 -> $AI_RESULTS/jedi/e25x_verify_<model>_trained.json
   python -m interp.e25x_verify --model {3f,16f} --part untr --seeds 10:25      -> e25x_verify_<model>_untr<a>_<b>.json

Fresh sample: fit = jedi_core.sample(2000, 2561), test = sample(2000, 2562) minus rows also in the fit sample (16 columns from
val_c16_f16, checked against sample() x as in E25x). Pairs: 3,000 train (rng 2563; last 20 % for selection), 3,000 test (rng 2564).
Edits on the base jet:
  M0, M2: E25w pair_edits (as E25x).
  Rr  (control; M0, M2 and every pT_i fixed, every pT_i r_i^2 changed): per-particle angle scale s_i = kappa r_i^(gamma - 1), so
      r_i -> kappa r_i^gamma; kappa = sqrt(M2 / sum pT r^(2 gamma)) keeps M2 = sum pT r^2 fixed; log gamma ~ U(log 0.5, log 2) per pair.
  Rot (control; every pT_i and r_i fixed): rotation of (deta, dphi) about the stored axis by U(-pi, pi).
Interventions: E25x design "don" (h' = clip(h_b + (h_don - h_b) Q Q^T), same Q on every particle at a), k = 1, E25x PNet / fit_pp /
  das_fit / eval_Q unchanged. Trained: a (M0, M2, Rr, Rot; 600 steps, minibatch 512; 100 random subspaces), a at the E25x baseline
  budget (250 steps, minibatch 384; M0, M2), u (M2), S (M0, M2, Rr, Rot). Untr (jedi_core untrained=True seeds as given, read-out d7
  refit by logistic regression on the fit sample; qualify acc > 0.30 and >= 3 calls): a (M0, M2) at the trained budget (600 / 512)
  and S (M0, M2).
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True, choices=['3f', '16f'])
ap.add_argument('--part', default='trained', choices=['trained', 'untr']); ap.add_argument('--seeds', default='10:25')
ap.add_argument('--smoke', action='store_true')
args = ap.parse_args()
_argv = sys.argv; sys.argv = [_argv[0], '--model', args.model] + (['--smoke'] if args.smoke else [])
import interp.e25x_p_to_s as X  # noqa: E402  (module-level argparse reads the sys.argv set above)
sys.argv = _argv
E = X.E
from interp.e19_common import wilson  # noqa: E402
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
SM = args.smoke
PER = 120 if SM else 2000
NTR, NTE = (500, 400) if SM else (3000, 3000)
STEPS = 60 if SM else 600
NRAND = 5 if SM else 100
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def load():
    from jetdistill.config import RESULTS, DATA
    from interp.jedi_core import sample
    y = np.asarray(np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')['y'])
    V = np.load(Path(DATA) / 'val_c16_f16.npy', mmap_mode='r'); out = {}; dmax = 0.0
    for nm, seed in (('fit', 2561), ('test', 2562)):
        idx = E.rows(y, PER, seed); x3, ys, _ = sample(PER, seed); assert np.array_equal(ys, y[idx])
        Xa = np.asarray(V[idx, :64]).astype(np.float32); d = float(np.abs(Xa[..., [5, 8, 11]] - x3).max()); dmax = max(dmax, d)
        if d > 1e-3: raise SystemExit(f'ABORT: 16-feature columns differ from sample() x ({nm}: {d})')
        out[nm] = (Xa, ys, idx)
    keep = ~np.isin(out['test'][2], out['fit'][2]); out['test'] = tuple(a[keep] for a in out['test'])
    old = np.concatenate([E.rows(y, 2000, 2551), E.rows(y, 2000, 2552)])
    ov = dict(fit_in_e25x=int(np.isin(out['fit'][2], old).sum()), test_in_e25x=int(np.isin(out['test'][2], old).sum()))
    log(f'fit N {len(out["fit"][0])}; test N {len(out["test"][0])} ({int((~keep).sum())} overlap removed); max col diff {dmax:.2g}; rows shared with E25x {ov}')
    return out, dmax, int((~keep).sum()), ov


def edit_rr(Xb, Ib, gamma):
    """per-particle angle scale s_i = kappa r_i^(gamma-1); M0, M2 = sum pT r^2 and pT_i unchanged (all 16 columns recomputed)"""
    C = E.C; Xd = Xb.astype(np.float64); rl = Ib['rl']; Ed = Xd.copy()
    pt, eta, phi, er, pr = (Xd[..., C[k]] for k in ('pt', 'eta', 'phi', 'etarel', 'phirel'))
    r = np.hypot(er, pr); ok = rl & (r > 0); ptm = np.where(rl, pt, 0.0)
    rg = np.where(ok, np.maximum(r, 1e-12) ** gamma[:, None], 0.0)
    kap = np.sqrt((ptm * r ** 2).sum(1) / np.maximum((ptm * rg ** 2).sum(1), 1e-30))
    s = np.where(ok, kap[:, None] * np.maximum(r, 1e-12) ** (gamma[:, None] - 1), 1.0)
    er2, pr2 = er * s, pr * s; eta2 = eta + (er2 - er); phi2 = phi + (pr2 - pr)
    F0 = E.cols(pt, eta, phi, er, pr, Ib['m2'], Ib['ptJ'], Ib['EJ']); F1 = E.cols(pt, eta2, phi2, er2, pr2, Ib['m2'], Ib['ptJ'], Ib['EJ'])
    for k in F0:
        d = F1[k] - F0[k]
        if k == 'phi': d = E.wrap(d)
        v = Xd[..., C[k]] + d
        if k == 'phi': v = E.wrap(v)
        Ed[..., C[k]] = np.where(rl, v, Xd[..., C[k]])
    for k in ('etarot', 'phirot', 'costhetarel'): Ed[..., C[k]] = np.where(rl, Xd[..., C[k]] * s, Xd[..., C[k]])
    return Ed.astype(np.float32)


def setup():
    D, dmax, nover, ov = load(); Xf, yf, _ = D['fit']; Xt, yt, _ = D['test']
    If, It = E.jinfo(Xf), E.jinfo(Xt)
    pairs = dict(train=E.make_pairs(np.arange(len(Xf)), NTR, 2563, If), test=E.make_pairs(np.arange(len(Xt)), NTE, 2564, It))
    PE, meta = {}, {}
    for arm, Xa, I, sd in (('train', Xf, If, 2565), ('test', Xt, It, 2566)):
        b, s = pairs[arm]; pe, meta[arm] = E.pair_edits(Xa, I, b, s); pe = {k: pe[k] for k in ('M0', 'M2')}
        rng = np.random.default_rng(sd); Ib = E.sub(I, b)
        gam = np.exp(rng.uniform(np.log(0.5), np.log(2.0), len(b))); pe['Rr'] = edit_rr(Xa[b], Ib, gam)
        pe['Rot'] = E.edit16(Xa[b], Ib, rot=rng.uniform(-np.pi, np.pi, len(b)))
        for V in ('M0', 'M2', 'Rr', 'Rot'):   # realised change of M0, M2 (deltaR column) per edit
            Ie = E.jinfo(pe[V]); meta[arm][f'{V}_dlogM0_absmax'] = float(np.abs(np.log(Ie['M0'] / Ib['M0'])).max())
            meta[arm][f'{V}_dlogM2_median_abs'] = float(np.median(np.abs(np.log(np.maximum(Ie['M2'], 1e-30) / Ib['M2']))))
            meta[arm][f'{V}_dlogM2_absmax'] = float(np.abs(np.log(np.maximum(Ie['M2'], 1e-30) / Ib['M2'])).max())
        PE[arm] = {V: X.SEL(v) for V, v in pe.items()}
    log('edit meta (test):', {k: round(v, 4) for k, v in meta['test'].items()})
    S = dict(fit=X.SEL(Xf), test=X.SEL(Xt), yf=yf, yt=yt, If=If, It=It, pairs=pairs, PE=PE)
    R = dict(sample=dict(fit=dict(per_class=PER, seed=2561, N=len(Xf)), test=dict(per_class=PER, seed=2562, N=len(Xt), removed_overlap=nover),
                         rows_shared_with_e25x=ov, max_col_diff=dmax, pairs=dict(train=NTR, test=NTE, rng=(2563, 2564))), edit_meta=meta, procedure=__doc__)
    return S, R


def stage_pair(pn, net, S, arm, V, st):
    Xa = S['fit'] if arm == 'train' else S['test']; b, _ = S['pairs'][arm]; xd = S['PE'][arm][V]
    if st == 'S': return None, net.run(Xa[b], ('S',))['S'], net.run(xd, ('S',))['S']
    rb = pn.reps(Xa[b], (st,)); rd = pn.reps(xd, (st,)); return X.dummy(len(b)), rb[st], rd[st]


def cell(net, pn, S, rg, lab, V, st, steps, bsize, nrand, seed=11):
    t1 = time.time(); btr, _ = S['pairs']['train']
    Ltr, Lte, base_tr, base_te = lab['train'][V], lab['test'][V], lab['train']['base'], lab['test']['base']
    ctx_tr, hb_tr, hd_tr = stage_pair(pn, net, S, 'train', V, st); ctx_te, hb_te, hd_te = stage_pair(pn, net, S, 'test', V, st)
    if st == 'S':
        Q, info = E.das_fit(net, 'S', hb_tr, hd_tr, Ltr, base_tr, 1, seed=seed, steps=steps)
        def calls(Q_): return E.eval_Q(net, 'S', hb_te, hd_te, Q_)  # noqa: E306
    else:
        Q, info = X.fit_pp(pn, st, ctx_tr, hb_tr, hd_tr, Ltr, base_tr, rg[st], 1, seed=seed, steps=steps, bsize=bsize)
        def calls(Q_): return pn.calls(st, ctx_te, hb_te, hd_te, Q_, rg[st])  # noqa: E306
    c = calls(Q); e = E.eff(c, Lte, base_te); br = X.boot_reps(c, Lte, base_te)
    r = dict(eff=e, eff_ci=[float(np.quantile(br, .025)), float(np.quantile(br, .975))] if len(br) else None, n_eff=int((Lte != base_te).sum()),
             iia_all=float((c == Lte).mean()), train=info, steps=steps, bsize=bsize, calls=c.astype(int).tolist())
    if nrand:
        rr = np.random.default_rng(100); d = hb_te.shape[-1]
        nv = np.array([E.eff(calls(E.rand_Q(d, 1, rr)), Lte, base_te) or 0.0 for _ in range(nrand)])
        r['random'] = dict(mean=float(nv.mean()), q975=float(np.quantile(nv, .975)), max=float(nv.max()), n=int(len(nv)))
    log(f'  {net.name} {st} {V} k1 steps {steps}/{bsize}: eff {e} {r["eff_ci"]} (n {r["n_eff"]}); rand q975 {r.get("random", {}).get("q975")}; '
        f'best step {info["best_step"]}; {time.time() - t1:.0f}s')
    return r


def labels(net, S):
    cf, ct = net.calls(S['fit']), net.calls(S['test']); lab = {}
    for arm in ('train', 'test'):
        b, s = S['pairs'][arm]; c_all = cf if arm == 'train' else ct
        lab[arm] = dict(base=c_all[b], **{V: net.calls(S['PE'][arm][V]) for V in S['PE'][arm]})
    return lab, ct


def boot_diff(c1, l1, c2, l2, base, nb=1000, seed=8):
    """paired bootstrap over pairs of eff(c1 vs l1) - eff(c2 vs l2); each eff over its own call-changing pairs"""
    rng = np.random.default_rng(seed); n = len(base); E1 = l1 != base; E2 = l2 != base; o1 = (c1 == l1) & E1; o2 = (c2 == l2) & E2; v = []
    for _ in range(nb):
        i = rng.integers(0, n, n); d1, d2 = E1[i].sum(), E2[i].sum()
        if d1 >= 10 and d2 >= 10: v.append(o1[i].sum() / d1 - o2[i].sum() / d2)
    v = np.asarray(v); diff = float(o1.sum() / max(E1.sum(), 1) - o2.sum() / max(E2.sum(), 1))
    return dict(diff=diff, ci=[float(np.quantile(v, .025)), float(np.quantile(v, .975))], p_two=float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()) + 1 / (len(v) + 1))))


def part_trained():
    from interp.jedi_core import Jedi
    S, R = setup(); JS = OUT / f'e25x_verify_{args.model}{"_smoke" if SM else ""}_trained.json'
    def save(): JS.write_text(json.dumps(E.clean(R), indent=1))  # noqa: E306
    j = Jedi(64, X.FAM); net = E.Net(j, name=args.model); pn = X.PNet(net); rg = pn.ranges(S['fit'])
    lab, ct = labels(net, S); M = R['trained'] = {}
    M['acc_test'] = float((ct == S['yt']).mean()); M['acc_test_ci'] = wilson(int((ct == S['yt']).sum()), len(ct))
    M['cf_change_frac_test'] = {V: float((lab['test'][V] != lab['test']['base']).mean()) for V in S['PE']['test']}
    log(f'acc test {M["acc_test"]:.4f} {M["acc_test_ci"]}; call changes {M["cf_change_frac_test"]}'); save()
    C = M['cells'] = {}
    for V in ('M2', 'M0', 'Rr', 'Rot'):
        C[f'a|{V}'] = cell(net, pn, S, rg, lab, V, 'a', STEPS, 512, NRAND); save()
        C[f'S|{V}'] = cell(net, pn, S, rg, lab, V, 'S', STEPS, None, NRAND); save()
    C['u|M2'] = cell(net, pn, S, rg, lab, 'M2', 'u', STEPS, 512, 0); save()
    for V in ('M0', 'M2'):
        C[f'a|{V}|budget250'] = cell(net, pn, S, rg, lab, V, 'a', 100 if SM else 250, 384, 0); save()
    base = lab['test']['base']; Dd = M['diffs'] = {}
    def c_(k): return np.asarray(C[k]['calls'])  # noqa: E306
    Dd['S-a|M2'] = boot_diff(c_('S|M2'), lab['test']['M2'], c_('a|M2'), lab['test']['M2'], base)
    Dd['S-a|M0'] = boot_diff(c_('S|M0'), lab['test']['M0'], c_('a|M0'), lab['test']['M0'], base)
    Dd['u-a|M2'] = boot_diff(c_('u|M2'), lab['test']['M2'], c_('a|M2'), lab['test']['M2'], base)
    Dd['a: M2-Rr'] = boot_diff(c_('a|M2'), lab['test']['M2'], c_('a|Rr'), lab['test']['Rr'], base)
    Dd['S: M2-Rr'] = boot_diff(c_('S|M2'), lab['test']['M2'], c_('S|Rr'), lab['test']['Rr'], base)
    if C['a|Rot']['eff'] is not None and C['S|Rot']['eff'] is not None:
        Dd['a: M2-Rot'] = boot_diff(c_('a|M2'), lab['test']['M2'], c_('a|Rot'), lab['test']['Rot'], base)
    for k, v in Dd.items(): log(f'  diff {k}: {v}')
    R['wall_s'] = time.time() - T0; save(); log('done')


def part_untr():
    import gc, resource
    from interp.jedi_core import Jedi
    a, b = (int(v) for v in args.seeds.split(':'))
    S, R = setup(); JS = OUT / f'e25x_verify_{args.model}{"_smoke" if SM else ""}_untr{a}_{b}.json'
    def save(): JS.write_text(json.dumps(E.clean(R), indent=1))  # noqa: E306
    j = Jedi(64, X.FAM); R['nets'] = {}
    for s in range(a, b):
        t1 = time.time(); name = f'untr{s}'
        ju = Jedi(64, X.FAM, untrained=True, seed=s); KB0 = {k: (np.asarray(v[0], np.float32), np.asarray(v[1], np.float32)) for k, v in ju.KB.items()}
        del ju; gc.collect()
        pre = E.Net(j, KB0, name=name + '_pre'); z = pre.run(S['fit'], ('z',))['z']
        KB1, nlive = E.refit_d7(pre, z, S['yf'], np.arange(len(S['yf']))); net = E.Net(j, KB1, name=name); del pre, z
        c = net.calls(S['test']); acc = float((c == S['yt']).mean()); nd = int(len(np.unique(c)))
        M = dict(acc_test=acc, n_distinct_calls=nd, d7_live_inputs=nlive, qualifies=bool(acc > X.QACC and nd >= 3))
        log(f'{name}: test acc {acc:.4f}; distinct calls {nd}; qualifies {M["qualifies"]}')
        if M['qualifies']:
            pn = X.PNet(net); rg = pn.ranges(S['fit']); lab, _ = labels(net, S); M['cells'] = {}
            M['cf_change_frac_test'] = {V: float((lab['test'][V] != lab['test']['base']).mean()) for V in ('M0', 'M2')}
            for V in ('M0', 'M2'):
                for st in ('a', 'S'):
                    r = cell(net, pn, S, rg, lab, V, st, STEPS, 512 if st == 'a' else None, 0); r.pop('calls'); M['cells'][f'{st}|{V}'] = r
        M['wall_s'] = time.time() - t1; R['nets'][name] = M; save()
        log(f'{name}: wall {M["wall_s"]:.0f}s; rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MiB')
    R['wall_s'] = time.time() - T0; save(); log('done')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    dict(trained=part_trained, untr=part_untr)[args.part]()
