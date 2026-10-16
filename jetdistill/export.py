"""Stand-alone Python files of a formula (plain Python, `math` only) and their check.

  <name>.py             particles -> observables -> 16 neurons as if-statements -> the network's last layer -> class.
                        The terms of one observable in a neuron are merged exactly into one piecewise-linear chain
                        (if Q < t1: … elif …), products are written as they are.
  <name>_normalized.py  the same function with every weight rescaled to read as "how much this matters" (below).
Both files are run jet by jet on test jets and must give the formula's class (check()).

Normalized weights: each term is measured by its average absolute size on the training jets (avg_k), so
  neuron j:  z_j = S_j · (c_j + Σ_k share_k · term_k / avg_k),  Σ_k |share_k| = 1   (share_k = a_k·avg_k / S_j)
  class c:   logit_c = B_c + T_c · Σ_j share_jc · h_j / avg_j,  Σ_j |share_jc| = 1  (h_j after the network's rounding)."""
import hashlib, importlib.util, json, os, pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from . import formula as F
from .config import CLASSES
from .network import round_wrap
from .observables import library, quantities_source, code_names

SIG = 7   # significant digits of printed numbers


def g(v):
    t = f'{float(v):.{SIG}g}'
    return t if ('e' in t or '.' in t or 'n' in t) else t + '.0'


def chains(neuron, ranges):
    """exact regrouping: per observable, all its one-observable terms as one piecewise-linear chain over its range.
    Returns dict(intercept, pieces=[dict(q, segments=[[lo, hi, slope, offset], …])], products=[terms])"""
    merged = {}
    for t in neuron['terms']:
        key = (t['q'], t['kind'], t.get('t'), t.get('q2'), t.get('kind2'), t.get('t2'))
        merged[key] = dict(merged[key], coef=merged[key]['coef'] + t['coef']) if key in merged else dict(t)
    c0 = neuron['intercept']; by, prods = {}, []
    for t in merged.values():
        if t['coef'] == 0: continue
        (prods if t.get('q2') else by.setdefault(t['q'], [])).append(t)
    pieces = []
    for q, ts in by.items():
        lo, hi = ranges[q]; edges = [lo] + sorted({t['t'] for t in ts if t['kind'] != 'lin' and lo < t['t'] < hi}) + [hi]; segs = []
        for a, b in zip(edges[:-1], edges[1:]):
            mid = .5 * (a + b); slope = off = 0.
            for t in ts:
                c = t['coef']
                if t['kind'] == 'lin': slope += c
                elif t['kind'] == 'gt' and mid > t['t']: slope += c; off -= c * t['t']
                elif t['kind'] == 'lt' and mid < t['t']: slope -= c; off += c * t['t']
            if segs and abs(segs[-1][2] - slope) <= 1e-9 * max(1, abs(slope)) and abs(segs[-1][3] - off) <= 1e-9 * max(1, abs(off)): segs[-1][1] = b
            else: segs.append([a, b, slope, off])
        if len(segs) == 1 and abs(segs[0][2]) < 1e-15: c0 += segs[0][3]; continue      # constant over the whole range
        pieces.append(dict(q=q, segments=segs))
    return dict(intercept=c0, pieces=pieces, products=prods)


def _factor(kind, t, x):
    return (None, x) if kind == 'lin' else ((f'{x} > {g(t)}', f'({x} - {g(t)})') if kind == 'gt' else (f'{x} < {g(t)}', f'({g(t)} - {x})'))


def neuron_code(ch):
    L = [f'    z = {g(ch["intercept"])}']
    for pc in ch['pieces']:
        x = 'Q.' + pc['q']; segs = pc['segments']
        for s, (lo, hi, sl, of) in enumerate(segs):
            if sl == 0 and of == 0: continue
            val = (f'{g(sl)} * {x}' + (f' + {g(of)}' if of > 0 else f' - {g(-of)}' if of < 0 else '')) if sl else g(of)
            if len(segs) == 1: L.append(f'    z += {val}'); continue
            cond = f'{x} < {g(hi)}' if s == 0 else (f'{x} >= {g(lo)}' if s == len(segs) - 1 else f'{g(lo)} <= {x} < {g(hi)}')
            L += [f'    if {cond}:', f'        z += {val}']
    for t in ch['products']:
        c1, f1 = _factor(t['kind'], t.get('t'), 'Q.' + t['q']); c2, f2 = _factor(t['kind2'], t.get('t2'), 'Q.' + t['q2'])
        conds = [c for c in (c1, c2) if c]; val = f'{g(t["coef"])} * {f1} * {f2}'
        L += [f'    if {" and ".join(conds)}:', f'        z += {val}'] if conds else [f'    z += {val}']
    return L


def header(title, n, used, stats):
    lib = library(n)
    return [f'"""JEDI-linear jet tagger, {n} particles, 3 features: {title}, as if-statements.', '',
            f'Input:  the {n} hardest particles of a jet, hardest first, each (pT [GeV], Δη, Δφ) relative to the jet axis;',
            '        empty slots have pT = 0.', 'Output: the class (g, q, W, Z or t), the 5 logits and the 5 probabilities.', '',
            '1. quantities():  physics quantities of the particles.',
            "2. jet_layer_4(): the 16 neurons of the network's last hidden layer, each max(0, z) with z built from if-statements.",
            "3. logits():      the network's own last layer: each neuron is rounded to the network's fixed-point grid",
            '                  (round to a multiple of 2^-f, then wrap modulo 2^i), multiplied by the weights, plus the biases.',
            '4. classify():    softmax of the logits; the class is the largest logit.', '', stats, '', 'Quantities:',
            *[f'  Q.{q:22s} {lib[q].desc}' for q in used], '"""']


def tail(n):
    ex = lambda v: str(v[:n] + [0.0] * max(0, n - 8))
    return ['def classify(pt, eta, phi):', '    s = logits(jet_layer_4(quantities(pt, eta, phi)))', '    m = max(s)', '    e = [math.exp(x - m) for x in s]',
            '    p = [x / sum(e) for x in e]', '    return CLASSES[s.index(m)], s, p', '', '', "if __name__ == '__main__':",
            f'    pt = {ex([412.0, 230.5, 101.2, 40.3, 22.8, 10.1, 6.4, 3.3])}', f'    eta = {ex([0.01, -0.12, 0.25, 0.05, -0.31, 0.2, -0.05, 0.4])}',
            f'    phi = {ex([-0.02, 0.18, -0.1, 0.33, 0.07, -0.25, 0.12, -0.36])}', '    c, s, p = classify(pt, eta, phi)', "    print('class:', c)",
            "    print('logits:', dict(zip(CLASSES, [round(x, 4) for x in s])))", "    print('probabilities:', dict(zip(CLASSES, [round(x, 4) for x in p])))", '']


def constants(last):
    K, b, i_, f_ = last
    return [f'CLASSES = {CLASSES!r}', f'W = {json.dumps([[float(v) for v in row] for row in K])}', f'B = {json.dumps([float(v) for v in b])}',
            f'INT_BITS = {json.dumps([int(v) for v in i_])}', f'FRAC_BITS = {json.dumps([int(v) for v in f_])}', '', '']


def write_formula(formula, n, last, ranges, title, stats, path):
    used = F.observables_used(formula)
    L = header(title, n, used, stats) + ['import math', 'from types import SimpleNamespace', ''] + constants(last) + [quantities_source(used, n), '', '']
    for nr in formula:
        L += [f'def neuron_{nr["neuron"]}(Q):'] + neuron_code(chains(nr, ranges)) + ['    return max(0.0, z)', '', '']
    L += ['def jet_layer_4(Q):', '    return [' + ', '.join(f'neuron_{j}(Q)' for j in (nr['neuron'] for nr in formula)) + ']', '', '',
          'def logits(h):', '    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]',
          '    return [B[c] + sum(h[j] * W[j][c] for j in range(len(h))) for c in range(5)]', '', ''] + tail(n)
    Path(path).parent.mkdir(parents=True, exist_ok=True); Path(path).write_text(code_names('\n'.join(L))); return Path(path)


# ---------------------------------------------------------------- normalized weights
def normalize(formula, Qtrain, last):
    K, b, i_, f_ = last; out = []
    for nr, B in zip(formula, F.bases(formula, Qtrain)):
        a = np.array([t['coef'] for t in nr['terms']]); avg = np.abs(B).mean(0); avg = np.where(avg > 0, avg, 1.0)
        S = float((np.abs(a) * avg).sum()) or 1.0; z = nr['intercept'] + (B @ a if len(a) else 0.0)
        h = round_wrap(np.maximum(z, 0), i_[nr['neuron']], f_[nr['neuron']])
        terms = sorted([dict(t, avg=float(v), share=float(c * v / S)) for t, c, v in zip(nr['terms'], a, avg)], key=lambda t: -abs(t['share']))
        out.append(dict(neuron=nr['neuron'], scale=S, c=nr['intercept'] / S, terms=terms, h_avg=float(np.mean(h)) or 1.0, on=float(np.mean(z > 0))))
    hav = np.array([nr['h_avg'] for nr in out]); V = K * hav[:, None]; T = np.abs(V).sum(0)
    return out, dict(T=T.tolist(), share=(V / T).tolist(), neuron_importance=(np.abs(V).sum(1) / np.abs(V).sum()).tolist(), h_avg=hav.tolist())


def _piece(q, k, t):
    return f'Q.{q}' if k == 'lin' else (f'max(0.0, Q.{q} - {g(t)})' if k == 'gt' else f'max(0.0, {g(t)} - Q.{q})')


def _readable(t):
    p = lambda q, k, th: q if k == 'lin' else (f'{q} > {th:.4g}' if k == 'gt' else f'{q} < {th:.4g}')
    return p(t['q'], t['kind'], t.get('t')) + (' and ' + p(t['q2'], t['kind2'], t.get('t2')) if t.get('q2') else '')


def write_normalized(formula, n, last, norm, title, stats, path):
    neurons, cls = norm; used = F.observables_used(formula)
    H = header(title + ', with normalized weights (how much each one matters)', n, used, stats)
    H[5:11] = ['Every weight is normalized so that it reads as "how much this matters"; the constants in front reproduce the formula.',
               '  neuron j:  z = S_j * (c_j + sum_k share_k * term_k / avg_k)   with  sum_k |share_k| = 1',
               "             avg_k = the term's average size on the training jets, so share_k is the fraction of the neuron's",
               '             average input that comes from that if-statement (sign: pushes it up / down).',
               '  class c:   logit_c = B_c + T_c * sum_j share_jc * h_j / avg_j   with  sum_j |share_jc| = 1', '',
               'How much each neuron matters (share of all class scores, averaged over the training jets):',
               *[f'  neuron {j:2d}: {100 * v:5.1f}%   (on for {100 * neurons[j]["on"]:.0f}% of jets)' for j, v in sorted(enumerate(cls['neuron_importance']), key=lambda x: -x[1])]]
    L = H + ['import math', 'from types import SimpleNamespace', ''] + constants(last) + [quantities_source(used, n), '', '']
    for nr in neurons:
        L += [f'def neuron_{nr["neuron"]}(Q):', f'    # scale S = {nr["scale"]:.4g}; each line: share * term / its average size', f'    z = {g(nr["scale"])} * ({g(nr["c"])}']
        for t in nr['terms']:
            ex = _piece(t['q'], t['kind'], t.get('t')) + (' * ' + _piece(t['q2'], t['kind2'], t.get('t2')) if t.get('q2') else '')
            L.append(f"        {'+' if t['share'] >= 0 else '-'} {g(abs(t['share']))} * {ex} / {g(t['avg'])}   # {100 * t['share']:+.1f}%  {_readable(t)}")
        L += ['    )', '    return max(0.0, z)', '', '']
    L += ['def jet_layer_4(Q):', '    return [' + ', '.join(f'neuron_{j}(Q)' for j in (nr['neuron'] for nr in neurons)) + ']', '', '',
          f'H_AVG = {json.dumps(cls["h_avg"])}', f'T = {json.dumps(cls["T"])}', '', '',
          'def logits(h):', '    h = [(math.floor(x * 2 ** f + 0.5) / 2 ** f) % 2 ** i for x, i, f in zip(h, INT_BITS, FRAC_BITS)]',
          '    # rounded to a multiple of 2^-20 (the formula\'s class scores are exact binary fractions): removes floating-point noise', '    return [round(v * 2 ** 20) / 2 ** 20 for v in [']
    for c in range(5):
        sh = sorted([(j, cls['share'][j][c]) for j in range(len(neurons)) if cls['share'][j][c] != 0], key=lambda x: -abs(x[1]))
        L.append(f'        {g(last[1][c])} + T[{c}] * (   # class {CLASSES[c]}')
        L += [f"            {'+' if s >= 0 else '-'} {g(abs(s))} * h[{j}] / H_AVG[{j}]" for j, s in sh] + ['        ),']
    L += ['    ]]', '', ''] + tail(n)
    Path(path).write_text(code_names('\n'.join(L))); return Path(path)


# ---------------------------------------------------------------- running a file on jets
def _load(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def _work(args):
    path, X = args; m = _load(path); return [m.classify(x[:, 0], x[:, 1], x[:, 2]) for x in X]


def run_file(path, x, workers=None):
    """[(class, logits, probabilities)] of the file for every jet of x (J, n, 3); parallel, cached by file content and jets"""
    path = Path(path); x = np.asarray(x, np.float32)
    key = hashlib.sha1(path.read_bytes() + x.tobytes()).hexdigest()[:20]; cache = path.parent / '.check_cache' / f'{path.stem}_{key}.pkl'
    if cache.exists(): return pickle.loads(cache.read_bytes())
    workers = workers or int(os.environ.get('CHECK_WORKERS', '6'))
    with ProcessPoolExecutor(workers) as ex: out = [r for o in ex.map(_work, [(str(path), p) for p in np.array_split(x, workers * 4)]) for r in o]
    cache.parent.mkdir(exist_ok=True); cache.write_bytes(pickle.dumps(out)); return out


def check(path, x, expected):
    """share of jets on which the file gives the expected class, and the file's logits"""
    R = run_file(path, x); cls = np.array([CLASSES.index(r[0]) for r in R])
    return float((cls == expected[:len(cls)]).mean()), np.array([r[1] for r in R])
