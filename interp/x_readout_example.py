"""What JEDI-linear itself "says" about the mass of the example W jet used in the explanatory figures (site/explain_figs.py).

The network has no mass output. Its mass reading is defined here as a linear read-out: a ridge regression from the 64 h4 units to
log(invariant mass), fitted on 10,000 ordinary test jets (seed 3; held-out R² printed). The same read-out is then applied to the
example jet (row 34 of the test file) shifted by δ, and with its particles measured from other axes; the network's class
probabilities are recorded too.  python -m interp.x_readout_example  ->  $AI_RESULTS/jedi/x_readout_example.json"""
import json, os, pathlib
import numpy as np
from sklearn.linear_model import RidgeCV
from interp.jedi_core import Jedi, sample
from jetdistill.config import RESULTS

OUT = pathlib.Path(os.environ.get('AI_RESULTS', 'results')) / 'jedi'; OUT.mkdir(parents=True, exist_ok=True)
net = Jedi(64)
x, y, Q = sample(per_class=2000, seed=3)
H = net.run(x, keep=('h4',))['h4']; t = np.log(Q['mass'] + 1e-3)
idx = np.random.default_rng(0).permutation(len(x)); tr, te = idx[: len(x) // 2], idx[len(x) // 2:]
rg = RidgeCV(alphas=np.logspace(-4, 3, 15)).fit(H[tr], t[tr]); r2 = rg.score(H[te], t[te])
print('read-out R2 (held out)', round(r2, 4), flush=True)

D = np.load(pathlib.Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')
J = 34; xj = np.asarray(D['x'][J], np.float32); real = xj[:, 0] > 0
pt, de, dp = xj[real, 0].astype(float), xj[real, 1].astype(float), xj[real, 2].astype(float)
CL = ['g', 'q', 'W', 'Z', 't']


def masses(de_, dp_):
    M0 = pt.sum(); M1 = np.array([(pt * de_).sum(), (pt * dp_).sum()]); M2 = (pt * (de_ ** 2 + dp_ ** 2)).sum()
    return float(np.sqrt(max(M0 * M2 - M1 @ M1, 0))), float(np.sqrt(M0 * M2))


def wta_axis(p, a, b):
    P = [(u, v, w) for u, v, w in zip(p, a, b)]
    while len(P) > 1:
        best = None
        for i in range(len(P)):
            for k in range(i + 1, len(P)):
                d = (P[i][1] - P[k][1]) ** 2 + (P[i][2] - P[k][2]) ** 2
                if best is None or d < best[0]: best = (d, i, k)
        _, i, k = best; A, B = P[i], P[k]; h = A if A[0] >= B[0] else B
        P = [P[n] for n in range(len(P)) if n not in (i, k)] + [(A[0] + B[0], h[1], h[2])]
    return np.array([P[0][1], P[0][2]])


def run(de_, dp_):
    xx = xj.copy(); xx[real, 1] = de_; xx[real, 2] = dp_
    r = net.run(xx[None], keep=('h4', 'logits')); lo = r['logits'][0]; p = np.exp(lo - lo.max()); p /= p.sum()
    return float(np.exp(rg.predict(r['h4'])[0])), {c: round(float(v), 4) for c, v in zip(CL, p)}


res = dict(row=J, readout_r2=r2, label=CL[int(D['y'][J])], shift=[], axes=[])
for d in np.round(np.linspace(0, 0.09, 31), 4):
    m, s = masses(de + d, dp); ro, pr = run(de + d, dp)
    res['shift'].append(dict(delta=float(d), m=m, sqrtM0M2=s, network_readout=ro, probs=pr, call=max(pr, key=pr.get)))
w = wta_axis(pt, de, dp); h = np.array([de[np.argmax(pt)], dp[np.argmax(pt)]])
for name, c in [('stored (E-scheme)', np.zeros(2)), ('winner-take-all', w), ('hardest particle', h)]:
    m, s = masses(de - c[0], dp - c[1]); ro, pr = run(de - c[0], dp - c[1])
    res['axes'].append(dict(axis=name, offset=[float(c[0]), float(c[1])], m=m, sqrtM0M2=s, network_readout=ro, probs=pr, call=max(pr, key=pr.get)))
(OUT / 'x_readout_example.json').write_text(json.dumps(res, indent=1))
for e in res['shift'][::5]: print(e['delta'], round(e['m'], 1), round(e['sqrtM0M2'], 1), round(e['network_readout'], 1), e['call'], e['probs'])
for e in res['axes']: print(e['axis'], round(e['m'], 1), round(e['sqrtM0M2'], 1), round(e['network_readout'], 1), e['call'], e['probs'])
