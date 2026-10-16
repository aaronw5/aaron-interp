"""The page of one setup and network: one HTML file (site/template.html with the data inlined) plus, next to it, the
explanation data of each formula (loaded when a formula is picked) and its Python files.
Tabs: Results, Neuron Interpretation, Class-Score Decomposition, Per-Jet Evaluation, Methodology, Python Implementation."""
import json, shutil, subprocess, tempfile
from pathlib import Path
import numpy as np
from .. import pipeline as P
from ..config import SETUPS, CLASSES, FIDELITY_LAMBDA, RESULTS
from ..network import Network
from ..observables import library, relabel, symbol, CODE_RENAME

TEMPLATE = Path(__file__).with_name('template.html')
SITE = RESULTS.parent / 'site'
FAMNAME = {('network', 'main'): 'network', ('network', 'other'): 'network', ('network', 'step4'): 'network-simplified',
           ('labels', 'main'): 'labels', ('labels', 'step4'): 'labels-simplified', ('neurons', 'main'): 'neurons', ('neurons', 'step4'): 'neurons-simplified'}


def rnd(o, keep=('coef', 't', 't2', 'intercept')):
    """floats to 4 significant digits (smaller page), except the formula's own numbers (the page computes with them)"""
    if isinstance(o, float): return float(f'{o:.4g}')
    if isinstance(o, list): return [rnd(v, keep) for v in o]
    if isinstance(o, dict): return {k: (v if k in keep else rnd(v, keep)) for k, v in o.items()}
    return o


def slim(full):
    """each group's worked example stores every observable's value once; fields the page does not use are dropped"""
    def walk(c):
        c.pop('pattern', None); c.pop('formula_right', None); x = c.get('example')
        if x and 'values' in x:
            qv = {}
            for v in x.pop('values'):
                qv[v['q']] = v['v']
                if v.get('q2'): qv[v['q2']] = v['v2']
            x['qv'] = qv
        for ch in c.get('children') or []: walk(ch)
    for n in full['anat']['neurons']:
        n.pop('regimes', None); n.pop('regime_r2', None)
        for c in n['combos']: walk(c)
    return full


def build(setup, n, site=SITE):
    base = P.out_dir(setup, n); out = Path(site) / setup / f'n{n}'; out.mkdir(parents=True, exist_ok=True)
    reg = P.formulas(setup, n); M = json.loads((base / 'metrics.json').read_text()); lib = library(n); S = SETUPS[setup]
    last = Network(n).last; K, b, i_, f_ = last
    fam = {t: FAMNAME[(m['family'], m['role'])] for t, m in reg.items()}; parent = {t: m['parent'] for t, m in reg.items() if m.get('parent')}
    models = [dict(r, family='network') if r['family'] == 'network' else dict(r, family='formula', kind='formula', title=r['name']) for r in M['models']]
    main = max((r for r in models if r.get('formula') and fam[r['formula']] == 'network'), key=lambda r: r['same_as_network'])['formula']
    D = dict(n=n, classes=CLASSES, mode='rosetta', fam=fam, parent=parent, main=main, metrics=dict(n_jets=M['n_jets'], models=models),
             exp=dict(name=setup, label=S.label, obs=S.page_desc, lam=FIDELITY_LAMBDA),
             last=dict(K7=K.tolist(), b7=b.tolist(), i7=i_.tolist(), f7=f_.tolist()))
    # the comparison table and the runs table (tuned on the network / on the labels, step 1)
    rows = [dict(M['models'][0], family='network')] + ([dict(M['step1'], family='step 1')] if M.get('step1') else [])
    rows += [dict(r, name=f"{r['formula']} ({r['name']})", family='labels' if reg[r['formula']]['family'] == 'labels' else 'network target')
             for r in M['models'][1:] if reg[r['formula']]['role'] != 'step4' and reg[r['formula']]['family'] in ('network', 'labels')]
    method = {}
    for t, m in reg.items():
        tj = base / 'formulas' / t / 'tuning.json'
        if m['family'] == 'network' and tj.exists():
            T = json.loads(tj.read_text()); method[str(T['K'])] = dict(step1=dict(terms=16 * T['K']), tuned=dict(terms=T['path'][0][0]), pruned=dict(terms=m['terms']))
    D['tunenet'] = dict(n_jets=M['n_jets'], rows=rows, method=method)
    # explanation data, Python files, formula structures
    D['explain'], D['py'], D['forms'] = {}, {}, {}
    for t, m in reg.items():
        d = base / 'formulas' / t
        if (d / 'anatomy.json').exists():
            A = json.loads((d / 'anatomy.json').read_text()); D['explain_rb'] = A['radial_bins']; D['explain_ptb'] = A.get('pt_bins')
            PK = json.loads((d / 'pack.json').read_text())
            keepN = ('values_by_true_class', 'auc_vs_rest', 'FACT_scale', 'FACT_used_by', 'mean_value_by_true_class', 'network_match'); keepS = ('cls', 'values_by_true_class', 'auc_vs_rest', 'FACT_scale')
            full = dict(anat=A, pack=dict(neurons={x['neuron']: {k: x[k] for k in keepN if k in x} for x in PK['neurons']}, class_scores=[{k: c[k] for k in keepS if k in c} for c in PK['class_scores']]),
                        expl=json.loads((d / 'explain.json').read_text()) if (d / 'explain.json').exists() else None)
            if (d / 'combos.json').exists(): full['combos'] = json.loads((d / 'combos.json').read_text())
            fn = f'explain_n{n}_{t}.json'; full = relabel(full, lib)   # quantity names shown as formula symbols
            (out / fn).write_text(json.dumps(rnd(slim(full)), separators=(',', ':'))); D['explain'][t] = dict(file=fn, has_text=full['expl'] is not None)
        if (d / 'check.json').exists():
            C = json.loads((d / 'check.json').read_text()); r = next(x for x in M['models'] if x.get('formula') == t)
            for suffix, key in (('', f'formula {t}'), ('_normalized', f'formula {t} normalized')):
                src = d / f'jedi_n{n}_formula{t}{suffix}.py'; shutil.copy2(src, out / src.name)
                D['py'][key] = dict(file=src.name, title=m['title'], terms=m['terms'], lines=len(src.read_text().splitlines()), n_jets=C['n_jets'],
                                    same_class_as_pipeline=C['same_class_file' if not suffix else 'same_class_normalized_file'], accuracy_file=r['accuracy'],
                                    accuracy_pipeline=r['accuracy'], agree_with_network_file=r['same_as_network'], network_accuracy=M['models'][0]['accuracy'])
        if (d / 'normalized.json').exists():
            N = json.loads((d / 'normalized.json').read_text()); D['forms'][t] = dict(norm=N, trees={}, big=[])
    D['fjets'] = example_jets(setup, n, reg, last)
    py = D.pop('py'); D = relabel(D, lib); D['py'] = py   # quantity names shown as formula symbols; the pop-up gives each definition
    D['defs'] = {symbol(k): f'{o.label}: {o.desc} (code name: {CODE_RENAME.get(k, k)})' for k, o in lib.items()}
    page = TEMPLATE.read_text().replace('/*DATA*/', json.dumps(D, separators=(',', ':')))
    js = page[page.rfind('<script>') + 8:page.rfind('</script>')]
    if shutil.which('node'):
        with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False) as tf: tf.write(js)
        r = subprocess.run(['node', '--check', tf.name], capture_output=True, text=True)
        if r.returncode: raise SystemExit('page not written: JavaScript syntax error\n' + r.stderr[-1500:])
    (out / 'index.html').write_text(page); return out / 'index.html'


def example_jets(setup, n, reg, last, per_class=20):
    """100 test jets (20 per true class) for the per-jet tab: particles, observables the formulas use, classes"""
    from .. import formula as F
    T = P.jets(n, 'full_test'); rng = np.random.default_rng(3)
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(T['y'] == c), per_class, replace=False) for c in range(5)]))
    fs = {t: P.load_formula(setup, n, t) for t in reg}; need = sorted({q for f in fs.values() for q in F.observables_used(f)})
    Q = {q: T['Q'][q][idx] for q in need}
    return dict(y=T['y'][idx].tolist(), net=T['net'][idx].tolist(), parts=[[[round(float(v), 4) for v in p] for p in x if p[0] > 0] for x in T['x'][idx]],
                q={q: [float(v) for v in Q[q]] for q in need}, formula={t: F.logits(f, Q, last).argmax(1).tolist() for t, f in fs.items()})
