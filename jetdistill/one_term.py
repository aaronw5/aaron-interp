"""At most one term per observable per neuron. Every neuron of a formula is a sum of terms; here the terms of one
observable (and, for products, of one pair of observables) are replaced by the single term that best reproduces their
sum: Q, max(0, Q − t) or max(0, t − Q) — t from the group's own thresholds and the observable's 5–95 % quantiles — or,
for a pair, the best of the group's own products (least squares on the fitting jets; the rest goes to the intercept).
Then the coefficients are re-tuned as in step 2 (tuning.train, toward the network's probabilities, λ·R) and the
formula is registered as family 'network', role 'other', parent = the formula it came from.

  python -m jetdistill.one_term SETUP N TAG [TAG ...]"""
import json, sys, time
import numpy as np
from . import config, formula as F, tuning
from .config import FIDELITY_LAMBDA
from .network import Network
from .pipeline import jets, sub, load_formula, add_formula, out_dir, log


def group_key(t):
    return (t['q'],) if not t.get('q2') else tuple(sorted((t['q'], t['q2'])))


def one_term(formula, Q):
    out = []
    for nr in formula:
        groups = {}
        for t in nr['terms']: groups.setdefault(group_key(t), []).append(t)
        terms, c0 = [], float(nr['intercept'])
        for key, ts in groups.items():
            if len(ts) == 1: terms.append(dict(ts[0])); continue
            y = sum(t['coef'] * F.basis(t, Q) for t in ts); ym = y.mean()
            if len(key) == 1:
                q = key[0]; v = np.asarray(Q[q], np.float64)
                ths = sorted({float(t['t']) for t in ts if t.get('t') is not None} | {float(x) for x in np.quantile(v, np.linspace(.05, .95, 19))})
                cands = [dict(q=q, kind='lin', t=None)] + [dict(q=q, kind=k, t=th) for th in ths for k in ('gt', 'lt')]
            else:
                cands = [{k: x for k, x in t.items() if k not in ('coef', 'importance')} for t in ts]
            best = None
            for c in cands:
                b = F.basis(c, Q); bm = b.mean(); bc = b - bm; den = float(bc @ bc)
                if den < 1e-12: continue
                a = float(bc @ (y - ym)) / den; r = (y - ym) - a * bc; err = float(r @ r)
                if best is None or err < best[0]: best = (err, c, a, ym - a * bm)
            if best is None: c0 += float(ym); continue
            terms.append(dict(best[1], coef=best[2])); c0 += float(best[3])
        out.append(dict(neuron=nr['neuron'], intercept=c0, terms=terms))
    return out


def agreement(f, J, last):
    return float((F.logits(f, J['Q'], last).argmax(1) == J['net']).mean())


def run(setup='all', n=8, tags=('805',)):
    last = Network(n).last; fit = sub(jets(n, 'fit', log), config.N_TUNE_FIT); dev = jets(n, 'dev', log); test = jets(n, 'full_test', log); res = {}
    for tag in tags:
        t0 = time.time(); f0 = load_formula(setup, n, tag); f1 = one_term(f0, fit['Q'])
        f2, s = tuning.train(f1, fit['Q'], dev['Q'], 'probabilities', fit, dev, last, lam=FIDELITY_LAMBDA)
        r = dict(start_terms=F.n_terms(f0), terms=F.n_terms(f2), ifs=F.n_ifs(f2), observables=len(F.observables_used(f2)),
                 test_start=agreement(f0, test, last), test_one_term_ls=agreement(f1, test, last), test=agreement(f2, test, last), dev=s,
                 test_accuracy=float((F.logits(f2, test['Q'], last).argmax(1) == test['y']).mean()))
        new = add_formula(setup, n, f2, 'network', 'other', f'one term per observable per neuron (from the {tag}), re-tuned', parent=tag)
        r['tag'] = new; res[tag] = r
        log(f"{setup} n{n} {tag} → {new}: {r['start_terms']} → {r['terms']} terms; same class as the network on the test file "
            f"{100 * r['test_start']:.2f}% → {100 * r['test_one_term_ls']:.2f}% (one term, least squares) → {100 * r['test']:.2f}% (re-tuned), accuracy {100 * r['test_accuracy']:.2f}%, {time.time() - t0:.0f} s")
    (out_dir(setup, n) / 'one_term.json').write_text(json.dumps(res, indent=1)); return res


if __name__ == '__main__':
    a = sys.argv[1:]; run(a[0], int(a[1]), a[2:] or ('805',))
