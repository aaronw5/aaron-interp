"""X9w verification (skeptical re-run on a fresh sample, plus controls the X9w run did not have).

Re-runs the X9w evaluation (interp.x9w_repr_2x2: same 33 networks + official JEDI-linear, same edit constructions) on
interp.jedi_core.sample(per_class=2000, seed=2610) (N = 10,000, 2,000 true W; overlap with the X9w sample(2000, 909) rows is
reported), and adds:
  T12    every real particle +0.12 rad in deta (outside the training augmentation range |c| <= 0.08): does the invariance of the
         augmented networks extend beyond the trained range?
  Tneg   -0.06 rad in deta; Tphi  +0.06 rad in dphi: direction dependence (JEDI aug seeds 0, 2 showed a +deta excess over Tr)
  Tr12   0.12 rad in a random direction per jet (rng 9094): equal-length null for T12
  W -> g/q flow (edited - unedited) for M2held (X9w quoted absolute W -> g/q fractions, not flows)
  mean total-variation distance of the softmax, TV = 0.5 sum_c |p_c(edit) - p_c(unedited)|, per edit, with jet-bootstrap CI
  (all 10,000 jets): does an augmented network's output move under T more than under the rotation null, beyond the W -> Z calls?
Seed level: stored vs aug for PFN/EFN/Deep Sets, exact stratified permutation (6 relabellings per architecture, 216 total);
JEDI-linear stored vs aug exact 3 vs 3 (interp.x9w_repr_2x2.perm2).
  ./job.sh x9w-verify 'python -m interp.x9w_verify'      -> $AI_RESULTS/jedi/x9w_verify.json
"""
import gc, itertools, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import x9w_repr_2x2 as X  # noqa: E402

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'
SEED = 2610; W_, Z_ = X.W_, X.Z_; T0 = time.time()


def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def softmax(lo):
    e = np.exp(lo - lo.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


def build(x):
    from interp.x9_efn_pfn import translate
    E, hyg = X.build_edits(x); J = len(x)
    E['T12'] = translate(x, 0.12, 0.0)
    E['Tneg'] = translate(x, 0.06, np.pi)
    E['Tphi'] = translate(x, 0.06, np.pi / 2)
    E['Tr12'] = translate(x, 0.12, np.random.default_rng(9094).uniform(0, 2 * np.pi, J))
    return E, hyg


def evaluate(fn, rc, x, y, E):
    prep = X.recentre if rc else (lambda v: v)
    W = y == W_; J = len(y)
    lo0 = fn(prep(x)); p0 = softmax(lo0); c0 = lo0.argmax(1); z0 = c0[W] == Z_; gq0 = np.isin(c0[W], [0, 1])
    o = dict(acc=float((c0 == y).mean()), acc_ci=X.wilson(int((c0 == y).sum()), J), W_to_Z_unedited=float(z0.mean()),
             W_to_gq_unedited=float(gq0.mean()), edits={})
    Zc, TV, CH = {}, {}, {}
    for k, xe in E.items():
        lo = fn(prep(xe)); ce = lo.argmax(1); ze = ce[W] == Z_; gq = np.isin(ce[W], [0, 1]); fl = ze.astype(float) - z0
        tv = 0.5 * np.abs(softmax(lo) - p0).sum(1); ch = ce != c0; Zc[k], TV[k], CH[k] = ze, tv, ch
        o['edits'][k] = dict(W_to_Z=float(ze.mean()), flow=float(fl.mean()), flow_ci=X.bci(fl),
                             flow_gq=float((gq.astype(float) - gq0).mean()), flow_gq_ci=X.bci(gq.astype(float) - gq0),
                             calls_changed=float(ch.mean()), tv_mean=float(tv.mean()), tv_ci=X.bci(tv),
                             max_abs_dlogit=float(np.abs(lo - lo0).max()), mcnemar_vs_unedited=X.mcnemar(ze, z0))
    for k in E:
        if k != 'rot':
            o['edits'][k]['mcnemar_Z_vs_rot'] = X.mcnemar(Zc[k], Zc['rot'])
            o['edits'][k]['mcnemar_calls_vs_rot'] = X.mcnemar(CH[k], CH['rot'])
            d = TV[k] - TV['rot']; o['edits'][k]['tv_minus_rot'] = float(d.mean()); o['edits'][k]['tv_minus_rot_ci'] = X.bci(d)
    for a, b in (('T', 'Tr'), ('T', 'Tneg'), ('T', 'Tphi'), ('T12', 'Tr12')):
        d = Zc[a].astype(float) - Zc[b]; o[f'flow_{a}_minus_{b}'] = float(d.mean()); o[f'flow_{a}_minus_{b}_ci'] = X.bci(d)
        o[f'mcnemar_{a}_vs_{b}'] = X.mcnemar(Zc[a], Zc[b])
    fT = Zc['T'].astype(float) - z0; fD = Zc['dilM2'].astype(float) - z0
    o['ratio_T_over_dilM2'] = float(fT.mean() / fD.mean()) if fD.mean() > 0 else None
    o['ratio_T_over_dilM2_ci'] = X.bratio(fT, fD) if fD.mean() > 0 else [None, None]
    return o


def strat_exact(V):
    """V: {arch: (stored values, aug values)}; statistic = mean over archs of mean(stored) - mean(aug); exact stratified relabelling"""
    archs = list(V); obs = np.mean([np.mean(V[a][0]) - np.mean(V[a][1]) for a in archs])
    opts = []
    for a in archs:
        v = np.asarray(list(V[a][0]) + list(V[a][1]), float); n = len(V[a][0]); o = []
        for S in itertools.combinations(range(len(v)), n):
            m = np.zeros(len(v), bool); m[list(S)] = True; o.append(v[m].mean() - v[~m].mean())
        opts.append(o)
    c = t = 0
    for combo in itertools.product(*opts):
        t += 1; c += abs(np.mean(combo)) >= abs(obs) - 1e-12
    return dict(effect=float(obs), p=c / t, n_relabellings=t)


def main():
    from interp.jedi_core import sample
    x, y, _ = sample(2000, SEED); x = np.ascontiguousarray(x, np.float32)
    ov = int(np.isin(X.sample_rows(2000, SEED), X.sample_rows(2000, X.SEED_SAMPLE)).sum())
    E, hyg = build(x)
    R = dict(header='see interp/x9w_verify.py docstring', sample=dict(per_class=2000, seed=SEED, N=len(y), N_W=int((y == W_).sum()),
             overlap_with_x9w_sample=ov), edit_hygiene=hyg, models={})
    log('overlap rows with sample(2000, 909):', ov)
    fo = OUT / 'x9w_verify.json'; save = lambda: fo.write_text(json.dumps(X.clean(R), indent=1))
    todo = [('official', 0, 'stored')] + [(a, s, c) for a in X.ARCHS for c in X.CELLS for s in X.seeds(a, c)]
    for a, s, c in todo:
        key = f'{a}_{c}_s{s}'
        try:
            fn, info = X.load_model(a, s, c)
        except Exception as e:  # noqa: BLE001
            log('SKIP', key, repr(e)[:300]); R['models'][key] = dict(error=repr(e)[:500]); continue
        rc = c.startswith('rc'); o = evaluate(fn, rc, x, y, E); o.update(arch=a, cell=c, seed=s, acc_test=info['acc_test']); R['models'][key] = o
        e = o['edits']
        log(f'{key}: acc {o["acc"]:.4f} WZ0 {o["W_to_Z_unedited"]:.3f} gq0 {o["W_to_gq_unedited"]:.3f} | ' +
            ' '.join(f'{k} {v["flow"]:+.3f}' for k, v in e.items()) + f' | M2held gq flow {e["M2held"]["flow_gq"]:+.3f}'
            f' | TV T {e["T"]["tv_mean"]:.3f} rot {e["rot"]["tv_mean"]:.3f} T12 {e["T12"]["tv_mean"]:.3f}')
        save(); del fn; gc.collect()
        if a in ('jedi', 'official'):
            import jax, keras; jax.clear_caches(); keras.backend.clear_session()
    keys = [k for k, v in R['models'].items() if 'edits' in v]
    for test, get in (('T_vs_rot', lambda v: v['edits']['T']['mcnemar_Z_vs_rot']['p']), ('T12_vs_rot', lambda v: v['edits']['T12']['mcnemar_Z_vs_rot']['p']),
                      ('T_vs_Tr', lambda v: v['mcnemar_T_vs_Tr']['p'])):
        ps = [get(R['models'][k]) for k in keys]; h = X.holm(ps)
        for k, p, hh in zip(keys, ps, h): R['models'][k].setdefault('holm', {})[test] = dict(p=p, p_holm=hh, m=len(keys))
    S = {}
    g = lambda a, c, ed: [R['models'][f'{a}_{c}_s{s}']['edits'][ed]['flow'] for s in X.seeds(a, c)]
    for ed in ('T', 'T12', 'Tr', 'M2held', 'dilM2'):
        S[f'stored_vs_aug_flow_{ed}_pfn_efn_dsets'] = strat_exact({a: (g(a, 'stored', ed), g(a, 'aug', ed)) for a in ('pfn', 'efn', 'dsets')})
        S[f'stored_vs_aug_flow_{ed}_jedi'] = X.perm2(g('jedi', 'stored', ed), g('jedi', 'aug', ed))
    R['seed_level'] = S; save()
    log('seed level', json.dumps(X.clean(S)))
    log('wrote', fo, 'X9w-verify DONE')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True); main()
