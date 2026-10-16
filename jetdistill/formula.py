"""A formula: the 16 neurons of the last hidden layer, each a sum of terms of observables, plus the network's last layer.

JSON form (a list, one entry per neuron):
  {"neuron": j, "intercept": c0, "terms": [{"q": <observable id>, "kind": "lin"|"gt"|"lt", "t": threshold, "coef": a,
                                          ["q2", "kind2", "t2" for a product of two]}, ...]}
  kind "lin": Q;  "gt": max(0, Q − t)  ("if Q > t: add a·(Q − t)");  "lt": max(0, t − Q).
Neuron j = max(0, c0 + Σ a·term); the class scores are the network's own last layer applied to the 16 neurons
(network.logits_from_h: round to the network's fixed-point grid, wrap, weights, biases)."""
import json
from pathlib import Path
import numpy as np
from .network import logits_from_h


def factor(x, kind, t):
    x = np.asarray(x, np.float64)
    if kind == 'lin': return x
    if kind == 'gt': return np.maximum(x - t, 0)
    if kind == 'lt': return np.maximum(t - x, 0)
    raise ValueError(kind)


def basis(term, Q):
    """the value of one term without its coefficient; Q = {observable id: values}"""
    b = factor(Q[term['q']], term['kind'], term.get('t'))
    return b * factor(Q[term['q2']], term['kind2'], term.get('t2')) if term.get('q2') else b


def bases(formula, Q):
    """per neuron, the (J, terms) matrix of its term values"""
    J = len(next(iter(Q.values())))
    return [np.stack([basis(t, Q) for t in nr['terms']], 1) if nr['terms'] else np.zeros((J, 0)) for nr in formula]


def pre_activations(formula, Q):
    """(J, 16): each neuron before its ReLU"""
    return np.stack([nr['intercept'] + B @ np.array([t['coef'] for t in nr['terms']], np.float64).reshape(-1) for nr, B in zip(formula, bases(formula, Q))], 1)


def hidden(formula, Q):
    return np.maximum(pre_activations(formula, Q), 0)


def logits(formula, Q, last):
    return logits_from_h(hidden(formula, Q), *last)


def n_terms(formula):
    return sum(len(nr['terms']) for nr in formula)


def n_ifs(formula):
    return sum((t['kind'] != 'lin') + (t.get('kind2', 'lin') != 'lin') for nr in formula for t in nr['terms'])


def observables_used(formula):
    return sorted({t['q'] for nr in formula for t in nr['terms']} | {t['q2'] for nr in formula for t in nr['terms'] if t.get('q2')})


def with_coefs(formula, params):
    """a copy with new (coefficients, intercept) per neuron"""
    return [dict(nr, intercept=float(c0), terms=[dict(t, coef=float(a)) for t, a in zip(nr['terms'], c)]) for nr, (c, c0) in zip(formula, params)]


def coefs(formula):
    return [(np.array([t['coef'] for t in nr['terms']], np.float64), float(nr['intercept'])) for nr in formula]


def load(path):
    return sorted(json.loads(Path(path).read_text()), key=lambda nr: nr['neuron'])


def save(formula, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([dict(neuron=nr['neuron'], intercept=nr['intercept'], terms=nr['terms']) for nr in formula], indent=1))


def tag(formula):
    """a formula's name on the pages: its number of terms"""
    return str(n_terms(formula))
