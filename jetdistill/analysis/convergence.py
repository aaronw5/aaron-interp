"""Step-2 convergence and overfitting check.

(a) The step-2 tuning of the network family (first K = 100 if-statements per neuron, λ = 0.01, toward the network's
    probabilities) run for 3000 Adam steps instead of the pipeline's steps; every 50 steps: training loss and agreement
    with the network on the training and on the validation jets.
(b) Every formula of the setup tuned on the network: agreement with the network on the training, validation and test jets
    (a formula that overfits agrees more on the jets it was tuned on).
Output: results/analysis/convergence_<setup>_n<N>.json
Usage: python -m jetdistill.analysis.convergence <N> [setup] [steps]"""
import json, sys
import numpy as np
from .. import pipeline as P, tuning, mars, formula as F
from ..config import RESULTS, SETUPS, FIDELITY_LAMBDA, N_TUNE_FIT
from ..network import Network


def run(n, setup='all', steps=3000, K=100):
    last = Network(n).last; fit = P.sub(P.jets(n, 'fit'), N_TUNE_FIT); dev = P.jets(n, 'dev'); test = P.jets(n, 'full_test')
    target = SETUPS[setup].target or 'probabilities'; hist = []
    f0 = tuning.refit([dict(nr, terms=nr['terms'][:K]) for nr in mars.load(setup, n)], fit['Q'], fit['Z'])
    tuning.train(f0, fit['Q'], dev['Q'], target, fit, dev, last, lam=FIDELITY_LAMBDA, steps=steps, history=hist)
    for h in hist[::10]: print(h, flush=True)
    final = {}
    for t, m in P.formulas(setup, n).items():
        if m['family'] != 'network': continue
        f = P.load_formula(setup, n, t)
        final[t] = {w: float((F.logits(f, J['Q'], last).argmax(1) == J['net']).mean()) for w, J in (('train', fit), ('val', dev), ('test', test))}
        print(t, final[t], flush=True)
    out = RESULTS / 'analysis'; out.mkdir(parents=True, exist_ok=True)
    (out / f'convergence_{setup}_n{n}.json').write_text(json.dumps(dict(K=K, lam=FIDELITY_LAMBDA, target=target, history=hist, final=final), indent=1))


if __name__ == '__main__':
    a = sys.argv[1:]; run(int(a[0]), *(a[1:2] or ['all']), *([int(a[2])] if len(a) > 2 else []))
