"""One setup, one network, end to end:   python -m jetdistill.pipeline <setup> <N> [stage ...]

Stages (default: all, in this order):
  step1    MARS on the network's neurons                                 results/<setup>/n<N>/step1.json
  tune     steps 2-3 for each family (below)                             results/<setup>/n<N>/formulas/<tag>/formula.json
  step4    smaller versions of each family's main formula                results/<setup>/n<N>/formulas/<tag>/ (+ step4_<family>.json)
  export   stand-alone Python files, checked on test jets                formulas/<tag>/jedi_n<N>_formula<tag>[_normalized].py
  explain  explanation data (pack with network match, anatomy)           formulas/<tag>/pack.json, anatomy.json
  metrics  every formula and the network on the whole test file         results/<setup>/n<N>/metrics.json
  page     the setup's page                                              site/<setup>/n<N>/index.html
Families: 'network' (tuned on the network's probabilities, or its decisions for all_agree; from 100 and 60 terms per
neuron), 'labels' (tuned on the true labels, from 60), 'neurons' (tuned on the network's neuron values, from 100). Each
formula is named by its number of terms (its tag). The written explanations (explain.json, combos.json) are added to
formulas/<tag>/ separately (see README) and picked up by the page.
Jets (observables and network outputs) are computed once per network and cached in results/_jets/n<N>/."""
import json, sys, time
import numpy as np
from . import config, data, formula as F, mars, tuning, simplify, export, metrics
from .config import SETUPS, RESULTS, FIDELITY_LAMBDA
from .network import Network
from .observables import compute, library


def log(*a):
    print(*a, flush=True)

FAMILIES = dict(network=((100, 60), None), labels=((60,), 'labels'), neurons=((100,), 'neurons')) if not config.SMOKE else \
           dict(network=((8, 5), None), labels=((5,), 'labels'), neurons=((8,), 'neurons'))
STEP4_METRIC = dict(network='agree', labels='acc', neurons='neuron')
TITLES = dict(network="tuned on the network's predictions", decisions="tuned on the network's decisions", labels='tuned on the true labels', neurons="tuned on the network's neuron values")


def out_dir(setup, n): return RESULTS / setup / f'n{n}'


# ---------------------------------------------------------------- cached jets
def jets(n, which, log=log):
    """dict(Q=observables, y=true class, net=network class, P=network probabilities, Z=network pre-activations, H=ReLU(Z));
    which: 'fit' (150,000 training jets), 'dev' (25,000), 'full_test' (the whole test file), 'explain' (60,000 training jets)"""
    f = RESULTS / '_jets' / f'n{n}' / f'{which}.npz'
    if not f.exists():
        t0 = time.time(); f.parent.mkdir(parents=True, exist_ok=True)
        if which == 'explain':
            X = np.load(config.DATA / 'train_ptetaphi_f16.npy', mmap_mode='r'); pool = np.setdiff1d(np.arange(len(X)), data.rows('dev'))
            rows = np.sort(pool[np.random.default_rng(0).choice(len(pool), config.N_EXPLAIN, replace=False)])
            x = np.asarray(X[rows, :n]).astype(np.float32)
            import h5py
            with h5py.File(config.DATA / 'train_meta.h5') as h: y = h['label'][:][rows]
        else:
            stop = config.N_STEP4_FIT if which == 'fit' else config.N_DEV if which == 'dev' else config.N_FULL_TEST; x = data.particles(which, n, stop=stop); y = data.labels(which, stop=stop)
        net = Network(n).run(x); Q = compute(x, n)
        np.savez(f, x=x, y=y.astype(np.int8), Z=net['z'].astype(np.float32), L=net['logits'].astype(np.float32), **{'Q_' + k: v.astype(np.float32) for k, v in Q.items()})
        log(f'jets {which}: {len(y)} in {time.time() - t0:.0f} s')
    D = np.load(f); L = D['L'].astype(np.float64); P = metrics.softmax(L); Z = D['Z'].astype(np.float64)
    return dict(x=D['x'], y=D['y'].astype(int), Q={k[2:]: D[k].astype(np.float64) for k in D.files if k.startswith('Q_')}, net=L.argmax(1), P=P, L=L, Z=Z, H=np.maximum(Z, 0))


def sub(J, stop):
    return {k: ({q: v[:stop] for q, v in J[k].items()} if k == 'Q' else J[k][:stop]) for k in J}


# ---------------------------------------------------------------- formulas on disk
def formulas(setup, n):
    """{tag: meta} of the formulas of a setup (meta: family, parent, title, file)"""
    f = out_dir(setup, n) / 'formulas.json'; return json.loads(f.read_text()) if f.exists() else {}


def add_formula(setup, n, f, family, role, title, parent=None):
    """register a formula; role: 'main' (the family's formula that step 4 starts from), 'other' or 'step4'"""
    reg = formulas(setup, n); base = F.tag(f); tag, k = base, 0
    while tag in reg: k += 1; tag = base + 'bcdefgh'[k - 1]                   # never two formulas with one name
    F.save(f, out_dir(setup, n) / 'formulas' / tag / 'formula.json')
    reg[tag] = dict(family=family, role=role, title=title, parent=parent, terms=F.n_terms(f), ifs=F.n_ifs(f), observables=len(F.observables_used(f)))
    (out_dir(setup, n) / 'formulas.json').write_text(json.dumps(reg, indent=1)); return tag


def drop_formulas(setup, n, keep):
    """remove the registered formulas for which keep(meta) is False (a stage that is rerun replaces its own formulas)"""
    import shutil
    reg = formulas(setup, n)
    for tag in [t for t, m in reg.items() if not keep(m)]:
        shutil.rmtree(out_dir(setup, n) / 'formulas' / tag, ignore_errors=True); reg.pop(tag)
    out_dir(setup, n).mkdir(parents=True, exist_ok=True); (out_dir(setup, n) / 'formulas.json').write_text(json.dumps(reg, indent=1))


def load_formula(setup, n, tag):
    return F.load(out_dir(setup, n) / 'formulas' / tag / 'formula.json')


# ---------------------------------------------------------------- stages
def stage_step1(setup, n, log=log):
    src = SETUPS[setup].step1_from or setup
    if not (out_dir(src, n) / 'step1.json').exists(): mars.run(src, n, log=log)


def stage_tune(setup, n, log=log):
    S = SETUPS[setup]; long = mars.load(setup, n); last = Network(n).last
    fit = sub(jets(n, 'fit', log), config.N_TUNE_FIT); dev = jets(n, 'dev', log); fams = dict(FAMILIES)
    fams['network'] = (fams['network'][0], S.target or 'probabilities')
    if S.truth_family is False: fams.pop('labels')
    drop_formulas(setup, n, lambda m: False)
    for fam, (Ks, target) in fams.items():
        for i, K in enumerate(Ks):
            f0 = tuning.refit([dict(nr, terms=nr['terms'][:K]) for nr in long], fit['Q'], fit['Z'])
            f, path = tuning.prune(f0, fit['Q'], dev['Q'], target, fit, dev, last, lam=0.0 if target == 'neurons' else FIDELITY_LAMBDA, log=log)
            title = TITLES['decisions' if target == 'decisions' else fam] + f' (from {K} if-statements per neuron, pruned)'
            tag = add_formula(setup, n, f, fam, 'main' if i == 0 else 'other', title)
            (out_dir(setup, n) / 'formulas' / tag / 'tuning.json').write_text(json.dumps(dict(K=K, target=target, lam=FIDELITY_LAMBDA if target != 'neurons' else 0, path=path)))
            log(f'{fam} K={K}: {tag}')


def main_formula(setup, n, family):
    return next((t for t, m in formulas(setup, n).items() if m['family'] == family and m['role'] == 'main'), None)


def stage_step4(setup, n, log=log):
    last = Network(n).last; J = dict(fit=jets(n, 'fit', log), dev=jets(n, 'dev', log))
    drop_formulas(setup, n, lambda m: m['role'] != 'step4')
    for fam in ('network', 'labels', 'neurons'):
        tag = main_formula(setup, n, fam)
        if tag is None: continue
        chosen, info = simplify.run(load_formula(setup, n, tag), J, last, STEP4_METRIC[fam], log=log)
        (out_dir(setup, n) / f'step4_{fam}.json').write_text(json.dumps(info, indent=1))
        if chosen: log(f"{fam}: smaller version {add_formula(setup, n, chosen, fam, 'step4', f'smaller version of the {tag} ({TITLES[fam]}; step 4)', parent=tag)}")


def stage_export(setup, n, check_jets=5000, log=log):
    last = Network(n).last; test = jets(n, 'full_test', log); fit = jets(n, 'fit', log)
    ranges = {q: (float(min(test['Q'][q].min(), fit['Q'][q].min())), float(max(test['Q'][q].max(), fit['Q'][q].max()))) for q in test['Q']}
    for tag, m in formulas(setup, n).items():
        f = load_formula(setup, n, tag); d = out_dir(setup, n) / 'formulas' / tag; L = F.logits(f, test['Q'], last); pred = L.argmax(1)
        stats = (f'Whole test file ({len(pred):,} jets): accuracy {100 * (pred == test["y"]).mean():.2f}% (the network: {100 * (test["net"] == test["y"]).mean():.2f}%); '
                 f'same class as the network for {100 * (pred == test["net"]).mean():.2f}% of jets.')
        norm = export.normalize(f, fit['Q'], last); (d / 'normalized.json').write_text(json.dumps(dict(neurons=norm[0], classes=norm[1])))
        p1 = export.write_formula(f, n, last, ranges, m['title'], stats, d / f'jedi_n{n}_formula{tag}.py')
        p2 = export.write_normalized(f, n, last, norm, m['title'], stats, d / f'jedi_n{n}_formula{tag}_normalized.py')
        s1, _ = export.check(p1, test['x'][:check_jets], pred); s2, _ = export.check(p2, test['x'][:check_jets], pred)
        (d / 'check.json').write_text(json.dumps(dict(n_jets=min(check_jets, len(pred)), same_class_file=s1, same_class_normalized_file=s2, lines=len(p1.read_text().splitlines()))))
        log(f'{tag}: files give the formula’s class on {100 * s1:.2f}% / {100 * s2:.2f}% of {check_jets} test jets')
        assert s1 == 1.0 and s2 == 1.0, f'exported files of {tag} differ from the formula'


def stage_explain(setup, n, log=log):
    from .explain import pack as P, anatomy as A
    last = Network(n).last; ex = jets(n, 'explain', log); dev = jets(n, 'dev', log); test = jets(n, 'full_test', log)
    for tag, m in formulas(setup, n).items():
        f = load_formula(setup, n, tag); d = out_dir(setup, n) / 'formulas' / tag; N = json.loads((d / 'normalized.json').read_text()); norm = (N['neurons'], N['classes'])
        pk = P.build(f, norm, n, last, ex, dev, m['title']); match = P.network_match(f, test['Q'], test['H'], last)
        for nr in pk['neurons']: nr['network_match'] = match[nr['neuron']]
        (d / 'pack.json').write_text(json.dumps(pk, indent=1))
        (d / 'anatomy.json').write_text(json.dumps(A.build(f, norm, n, last, ex, ex['x']), separators=(',', ':')))
        if (d / 'explain.json').exists():
            (d / 'combos_pack.json').write_text(json.dumps(P.combos_pack(json.loads((d / 'anatomy.json').read_text()), json.loads((d / 'explain.json').read_text()), pk, last), indent=1))
        log(f'{tag}: explanation data written')


def stage_metrics(setup, n, log=log):
    last = Network(n).last; test = jets(n, 'full_test', log)
    rows = [dict(family='network', name='the network', **metrics.metrics(test['L'], test['y'], test['net']))]
    for tag, m in formulas(setup, n).items():
        rows.append(dict(family=m['family'], formula=tag, name=m['title'], terms=m['terms'], parent=m.get('parent'),
                         **metrics.metrics(F.logits(load_formula(setup, n, tag), test['Q'], last), test['y'], test['net'])))
    long = mars.load(setup, n); fit = sub(jets(n, 'fit', log), config.N_TUNE_FIT)          # step 1 alone: 100 terms per neuron, least squares
    step1 = tuning.refit([dict(nr, terms=nr['terms'][:FAMILIES['network'][0][0]]) for nr in long], fit['Q'], fit['Z'])
    s1 = dict(name=f'{F.n_terms(step1)} (step 1: if-statements fitted to each neuron separately)', terms=F.n_terms(step1), **metrics.metrics(F.logits(step1, test['Q'], last), test['y'], test['net']))
    (out_dir(setup, n) / 'metrics.json').write_text(json.dumps(dict(n_jets=int(len(test['y'])), models=rows, step1=s1)))
    for r in rows: log(f"{r['name'][:70]:70s} accuracy {100 * r['accuracy']:.2f}%  same as the network {100 * r['same_as_network']:.2f}%")


def stage_page(setup, n, log=log):
    from .site import page
    log(f'page: {page.build(setup, n)}')


STAGES = dict(step1=stage_step1, tune=stage_tune, step4=stage_step4, export=stage_export, explain=stage_explain, metrics=stage_metrics, page=stage_page)


if __name__ == '__main__':
    setup, n, *todo = sys.argv[1:]; n = int(n)
    for s in todo or list(STAGES):
        t0 = time.time(); STAGES[s](setup, n); print(f'== {setup} n{n} {s} done in {time.time() - t0:.0f} s', flush=True)
