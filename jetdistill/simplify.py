"""Step 4 (optional): smaller versions of a tuned formula.

1. Candidate pool per neuron: its own terms plus, for every observable it uses, the observable itself and threshold terms
   at its 10/25/50/75/90 % quantiles.
2. Backward elimination per neuron on a weighted least-squares fit to the reference neuron values (rank-one downdates),
   then one global greedy order over all neurons (remove the term whose removal costs least).
   Reference and weights:  formulas tuned on the network or the labels — the START formula's own neurons, each jet
   weighted by how much the class probabilities depend on that neuron (Fisher information of the softmax through the
   last layer; jets where the neuron is off get 0.1 of its mean weight);  formulas tuned on the neuron values — the
   network's own neurons, weight 1/Var.
3. Whole observables are removed in rounds (10 at a time, the ones whose removal costs least at 250 terms), 3 rounds.
4. For each budget (600 … 120 terms) and pool round 0 or 1: assemble, clean up (always-on thresholds → linear, merge
   duplicates), refit one-sided (jets where the reference neuron is off only penalise a positive value), train
   coefficients and thresholds together (300 Adam steps), refit, round thresholds to 2 or 3 significant digits.
5. Choice: the smallest candidate whose validation score is within TOL of the START formula's (agreement with the
   network 0.5 points; accuracy 0.4 points; mean neuron R² 0.01).
Only training jets are used for fitting and validation jets for the choice."""
import copy
import numpy as np
from . import formula as F
from .network import round_wrap

TOL = dict(agree=0.005, acc=0.004, neuron=0.01)
KC = {'lin': 0, 'gt': 1, 'lt': 2}
from .config import SMOKE


def log(*a):
    print(*a, flush=True)
BUDGETS = (600, 500, 400, 350, 300, 250, 200, 150, 120) if not SMOKE else (60, 40)


class Step4:
    """jets: {'fit': J, 'dev': J} with J = dict(Q=observables, y=true class, net=network class, Z=network pre-activations);
    metric: 'agree' (tuned on the network), 'acc' (tuned on the labels) or 'neuron' (tuned on the neuron values)"""

    def __init__(self, start, jets, last, metric):
        self.start, self.jets, self.last, self.metric = start, jets, last, metric

    # ---------------- evaluation
    def zs(self, f, split):
        return list(F.pre_activations(f, self.jets[split]['Q']).T)

    def logits_from_z(self, Z):
        K, b, i, fb = self.last; return round_wrap(np.maximum(np.stack(Z, 1), 0), i, fb) @ K + b

    def score(self, f, split='dev'):
        J = self.jets[split]
        if self.metric == 'neuron':
            Hn = np.maximum(J['Z'], 0); H = np.maximum(np.stack(self.zs(f, split), 1), 0); v = Hn.var(0); ok = v > 1e-9
            return float(np.mean(1 - ((H - Hn) ** 2).mean(0)[ok] / v[ok]))
        pred = self.logits_from_z(self.zs(f, split)).argmax(1)
        return float((pred == (J['net'] if self.metric == 'agree' else J['y'])).mean())

    # ---------------- reference neuron values and weights
    def ref_z(self, split='fit'):
        return list(self.jets[split]['Z'].T) if self.metric == 'neuron' else self.zs(self.start, split)

    def _fisher(self, split):
        Z = self.zs(self.start, split); l = self.logits_from_z(Z); p = np.exp(l - l.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        K = self.last[0]; return Z, [(p * K[j] ** 2).sum(1) - (p @ K[j]) ** 2 for j in range(len(Z))]

    def weights(self, neg=0.1, split='fit'):
        """per neuron, per jet: the weight of the pool fits"""
        if self.metric == 'neuron':
            return [np.where(z > 0, 1.0, neg) / (np.maximum(z, 0).var() + 1e-9) for z in self.jets[split]['Z'].T]
        Z, W = self._fisher(split); out = []
        for z, w in zip(Z, W):
            on = z > 0; m = w[on].mean() if on.any() else 1e-6; out.append(np.where(on, w, neg * m))
        return out

    def weights_raw(self, split='fit'):
        if self.metric == 'neuron':
            return [np.full(len(z), 1.0 / (np.maximum(z, 0).var() + 1e-9)) for z in self.jets[split]['Z'].T]
        return [w + 1e-6 for w in self._fisher(split)[1]]

    # ---------------- pools and backward paths
    def pool_for(self, nr, qset, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
        D = self.jets['fit']['Q']; P = [dict(t) for t in nr['terms']]
        for q in sorted(qset):
            P.append(dict(q=q, kind='lin', t=None, coef=0.0))
            for v in np.unique(np.quantile(D[q], qs)): P += [dict(q=q, kind='gt', t=float(v), coef=0.0), dict(q=q, kind='lt', t=float(v), coef=0.0)]
        return P

    def build_paths(self, qset, W):
        D = self.jets['fit']['Q']; Zr = self.ref_z(); out = {}
        for j, nr in enumerate(self.start):
            nr2 = dict(nr, terms=[t for t in nr['terms'] if t['q'] in qset and (not t.get('q2') or t['q2'] in qset)])
            P = self.pool_for(nr2, qset); G, r, yy, sd = gram(P, D, W[j], Zr[j]); full, path = backward_rank1(G, r, yy)
            out[j] = dict(pool=P, G=G, r=r, yy=yy, sd=sd, full=full, path=path)
        return out

    def pools(self, rounds=3, batch=10, ref_budget=250 if not SMOKE else 40, log=log):
        """the pool snapshots of rounds 0..rounds (each round removes the `batch` cheapest observables)"""
        W = self.weights(); qset = set(F.observables_used(self.start)); PP = self.build_paths(qset, W); snaps = [PP]
        for rd in range(rounds):
            c = quantity_costs(PP, ref_budget); drop = list(c)[:batch]; log(f'pool round {rd + 1}: removing {drop}')
            qset = {q for q in c if q not in drop}; PP = self.build_paths(qset, W); snaps.append(PP)
        return snaps

    # ---------------- one candidate
    def refit_onesided(self, f, Wr, iters=6, ridge=1e-9, init_neg=0.1):
        """weighted LS with the ReLU-aware one-sided loss: reference z > 0: (z − z_ref)²; z_ref ≤ 0: max(0, z)² (IRLS)"""
        D = self.jets['fit']['Q']; Zr = self.ref_z(); out = copy.deepcopy(f)
        for j, nr in enumerate(out):
            if not nr['terms']: continue
            X = np.stack([F.basis(t, D) for t in nr['terms']] + [np.ones(len(Zr[j]))], 1)
            sd = X.std(0); sd[-1] = 1; sd[sd == 0] = 1; Xs = X / sd
            on = Zr[j] > 0; w = np.where(on, Wr[j], init_neg * Wr[j]); yt = Zr[j]
            for _ in range(iters):
                Xw = Xs * w[:, None]; G = Xw.T @ Xs
                beta = np.linalg.solve(G + ridge * np.trace(G) * np.eye(X.shape[1]), Xw.T @ yt)
                w = np.where(on | (Xs @ beta > 0), Wr[j], 0.0); yt = np.where(on, Zr[j], 0.0)
            beta = beta / sd
            for t, b in zip(nr['terms'], beta[:-1]): t['coef'] = float(b)
            nr['intercept'] = float(beta[-1])
        return out

    def refine_t(self, f, Wr, steps=300, lr=1e-3, lr_t=3e-3):
        """coefficients, intercepts and thresholds of all neurons trained together on the weighted one-sided loss"""
        import jax, jax.numpy as jnp
        D = self.jets['fit']['Q']; Zr = np.stack(self.ref_z(), 1).astype(np.float32); Wm = np.stack(Wr, 1).astype(np.float32)
        qs = F.observables_used(f); qi = {q: i for i, q in enumerate(qs)}
        X = jnp.asarray(np.stack([D[q] for q in qs], 1).astype(np.float32)); qsd = np.array([D[q].std() + 1e-12 for q in qs], np.float32)
        flat = [(j, t) for j, nr in enumerate(f) for t in nr['terms']]
        nj = jnp.asarray(np.array([j for j, _ in flat], np.int32)); q1 = np.array([qi[t['q']] for _, t in flat]); q2 = np.array([qi[t['q2']] if t.get('q2') else 0 for _, t in flat])
        k1 = jnp.asarray(np.array([KC[t['kind']] for _, t in flat])); k2 = jnp.asarray(np.array([KC[t['kind2']] if t.get('q2') else 0 for _, t in flat]))
        t1 = jnp.asarray(np.array([t.get('t') or 0.0 for _, t in flat], np.float32)); t2 = jnp.asarray(np.array([(t.get('t2') or 0.0) if t.get('q2') else 0.0 for _, t in flat], np.float32))
        has2 = jnp.asarray(np.array([bool(t.get('q2')) for _, t in flat])); s1 = jnp.asarray(qsd[q1]); s2 = jnp.asarray(qsd[q2])
        fac = lambda x, k, t: jnp.where(k == 0, x, jnp.where(k == 1, jax.nn.relu(x - t), jax.nn.relu(t - x)))
        Xq1, Xq2 = X[:, jnp.asarray(q1)], X[:, jnp.asarray(q2)]
        basis = lambda p: fac(Xq1, k1, t1 + p['d1'] * s1) * jnp.where(has2, fac(Xq2, k2, t2 + p['d2'] * s2), 1.0)
        bstd = jnp.asarray(np.asarray(basis(dict(d1=jnp.zeros_like(t1), d2=jnp.zeros_like(t2)))).std(0) + 1e-9)
        Zrj, Wj = jnp.asarray(Zr), jnp.asarray(Wm); on = Zrj > 0

        def loss(p):
            z = jax.ops.segment_sum((basis(p) * (p['c'] / bstd)).T, nj, num_segments=len(f)).T + p['b']
            e = jnp.where(on, z - Zrj, jax.nn.relu(z)); return 0.5 * jnp.mean((Wj * e * e).sum(1))
        p = dict(c=jnp.asarray(np.array([t['coef'] for _, t in flat], np.float32)) * bstd, b=jnp.asarray(np.array([nr['intercept'] for nr in f], np.float32)),
                 d1=jnp.zeros_like(t1), d2=jnp.zeros_like(t2))
        g = jax.jit(jax.value_and_grad(loss)); tm = jax.tree_util.tree_map; m_, v_ = tm(jnp.zeros_like, p), tm(jnp.zeros_like, p); lrs = dict(c=lr, b=lr, d1=lr_t, d2=lr_t)
        for i in range(steps):
            _, gr = g(p); m_ = tm(lambda a, b: .9 * a + .1 * b, m_, gr); v_ = tm(lambda a, b: .999 * a + .001 * b * b, v_, gr)
            p = {k: p[k] - lrs[k] * (m_[k] / (1 - .9 ** (i + 1))) / (jnp.sqrt(v_[k] / (1 - .999 ** (i + 1))) + 1e-8) for k in p}
        c = np.asarray(p['c'] / bstd, np.float64); d1 = np.asarray(p['d1'] * s1); d2 = np.asarray(p['d2'] * s2); out = []; i = 0
        for j, nr in enumerate(f):
            ts = []
            for t in nr['terms']:
                t = dict(t, coef=float(c[i]))
                if t['kind'] != 'lin': t['t'] = float(t['t'] + d1[i])
                if t.get('q2') and t['kind2'] != 'lin': t['t2'] = float(t['t2'] + d2[i])
                ts.append(t); i += 1
            out.append(dict(neuron=nr['neuron'], intercept=float(p['b'][j]), terms=ts))
        return out

    def canon(self, f):
        """always-on threshold terms → linear; never-on or zero terms dropped; identical terms merged"""
        D = self.jets['fit']['Q']; out = []
        for nr in f:
            b = nr['intercept']; acc = {}
            for t in nr['terms']:
                t = dict(t)
                if not t.get('q2') and t['kind'] != 'lin' and ((D[t['q']] > t['t']) if t['kind'] == 'gt' else (D[t['q']] < t['t'])).all():
                    s = 1 if t['kind'] == 'gt' else -1; b -= t['coef'] * s * t['t']; t = dict(q=t['q'], kind='lin', t=None, coef=t['coef'] * s)
                if t['coef'] == 0 or np.abs(F.basis(t, D)).max() == 0: continue
                key = (t['q'], t['kind'], t.get('t'), t.get('q2'), t.get('kind2'), t.get('t2'))
                if key in acc: acc[key]['coef'] += t['coef']
                else: acc[key] = t
            out.append(dict(neuron=nr['neuron'], intercept=b, terms=list(acc.values())))
        return out

    def candidate(self, PP, n_keep, steps=300 if not SMOKE else 30):
        """one budget from one pool snapshot -> {digits: formula}"""
        Wr = self.weights_raw()
        g = self.refit_onesided(self.canon(self.refit_onesided(assemble(PP, n_keep), Wr)), Wr)
        if steps: g = self.refit_onesided(self.canon(self.refine_t(g, Wr, steps=steps)), Wr)
        return {nd: sort_formula(round_coefs(self.refit_onesided(self.canon(round_thresholds(g, nd)), Wr), 3)) for nd in (2, 3)}

    def choose(self, candidates):
        """smallest candidate within TOL of the START formula's validation score (ties: higher score)"""
        ref = self.score(self.start); rows = sorted((F.n_terms(f), -self.score(f), i) for i, f in enumerate(candidates))
        ok = [r for r in rows if -r[1] >= ref - TOL[self.metric]]
        return (candidates[ok[0][2]] if ok else None), dict(reference=ref, tolerance=TOL[self.metric], candidates=[dict(terms=t, score=-s) for t, s, _ in rows])


# ---------------------------------------------------------------- linear algebra of the pools
def gram(terms, D, w, y, chunk=30000):
    n, p = len(y), len(terms); X = np.empty((n, p + 1), np.float32)
    for a, t in enumerate(terms): X[:, a] = F.basis(t, D)
    X[:, p] = 1
    mu = (w[:, None] * X).sum(0) / w.sum(); sd = np.sqrt((w[:, None] * (X - mu) ** 2).sum(0) / w.sum()); sd[p] = 1; sd[sd < 1e-12] = np.inf
    X /= sd.astype(np.float32); G = np.zeros((p + 1, p + 1)); r = np.zeros(p + 1)
    for s in range(0, n, chunk):
        Xc = X[s:s + chunk].astype(np.float64); wc = w[s:s + chunk]; G += (Xc * wc[:, None]).T @ Xc; r += (Xc * wc[:, None]).T @ y[s:s + chunk]
    return G / n, r / n, (w * y * y).sum() / n, sd


def backward_rank1(G, r, yy, ridge=1e-7):
    """backward elimination with rank-one inverse downdates (the last column, the intercept, stays).
    Returns the full fit's error and the path [(removed column, error after)]."""
    p = G.shape[0]; S = list(range(p))
    Gm = np.nan_to_num(G, nan=0.0, posinf=0.0); Gi = np.linalg.inv(Gm + ridge * np.eye(p) * max(np.trace(Gm) / p, 1e-12))
    sse = lambda Gi, S: (lambda beta: (0.5 * (yy - r[S] @ beta), beta))(Gi @ r[S])
    full, _ = sse(Gi, S); path = []
    while len(S) > 1:
        s0, beta = sse(Gi, S); d = np.diag(Gi); cost = np.array([0.5 * beta[k] ** 2 / d[k] if S[k] != p - 1 else np.inf for k in range(len(S))])
        k = int(np.argmin(cost)); a = S[k]
        Gi = Gi - np.outer(Gi[:, k], Gi[k, :]) / Gi[k, k]; Gi = np.delete(np.delete(Gi, k, 0), k, 1); S.pop(k); path.append((a, s0 + cost[k]))
    return full, path


def solve(G, r, S, ridge=1e-9):
    Gs = G[np.ix_(S, S)]; return np.linalg.solve(Gs + ridge * np.eye(len(S)) * max(np.trace(Gs) / len(S), 1e-12), r[S])


def kept_sets(PP, n_keep):
    """global greedy over the per-neuron paths: the kept pool columns of each neuron when n_keep terms remain in total"""
    ptr = {j: 0 for j in PP}; total = sum(len(PP[j]['path']) for j in PP)
    def cost(j):
        pth, k = PP[j]['path'], ptr[j]
        return np.inf if k >= len(pth) else pth[k][1] - (pth[k - 1][1] if k else PP[j]['full'])
    for _ in range(total - n_keep):
        _, j = min((cost(j), j) for j in PP); ptr[j] += 1
    return {j: [a for a in range(len(PP[j]['pool'])) if a not in {b for b, _ in PP[j]['path'][:ptr[j]]}] + [len(PP[j]['pool'])] for j in PP}


def assemble(PP, n_keep):
    f = []
    for j, S in sorted(kept_sets(PP, n_keep).items()):
        d = PP[j]; coef = solve(d['G'], d['r'], S) / d['sd'][S]
        f.append(dict(neuron=j, intercept=float(coef[-1]), terms=[dict(d['pool'][a], coef=float(c)) for a, c in zip(S[:-1], coef[:-1])]))
    return f


def quantity_costs(PP, n_keep):
    """per observable: the error added by deleting every kept term that uses it (cheapest first)"""
    cost = {}
    for j, S in kept_sets(PP, n_keep).items():
        d = PP[j]; sse = lambda S: 0.5 * (d['yy'] - d['r'][S] @ solve(d['G'], d['r'], S)); base = sse(S)
        for q in {d['pool'][a]['q'] for a in S[:-1]} | {d['pool'][a]['q2'] for a in S[:-1] if d['pool'][a].get('q2')}:
            cost[q] = cost.get(q, 0) + sse([a for a in S[:-1] if q not in (d['pool'][a]['q'], d['pool'][a].get('q2'))] + [S[-1]]) - base
    return dict(sorted(cost.items(), key=lambda kv: kv[1]))


# ---------------------------------------------------------------- rounding
def sig(x, n=2):
    if x is None or x == 0: return x
    from math import floor, log10
    return float(round(x, -int(floor(log10(abs(x)))) + n - 1))


def round_thresholds(f, n=2):
    g = copy.deepcopy(f)
    for nr in g:
        for t in nr['terms']:
            if t['kind'] != 'lin': t['t'] = sig(t['t'], n)
            if t.get('q2') and t['kind2'] != 'lin': t['t2'] = sig(t['t2'], n)
    return g


def round_coefs(f, n=3):
    return [dict(nr, intercept=sig(nr['intercept'], n), terms=[dict(t, coef=sig(t['coef'], n)) for t in nr['terms']]) for nr in f]


def sort_formula(f):
    return [dict(neuron=nr['neuron'], intercept=nr['intercept'], terms=sorted(nr['terms'], key=lambda t: (bool(t.get('q2')), t['q'], t.get('q2') or '', KC[t['kind']], t.get('t') or 0, t.get('t2') or 0))) for nr in f]


def run(start, jets, last, metric, total_budgets=BUDGETS, snapshots=(0, 1), log=log):
    """all candidates (budgets below the START size, pool rounds 0 and 1, 2 and 3 significant digits) and the choice"""
    S = Step4(start, jets, last, metric); PPs = S.pools(log=log); cands = []
    for r in snapshots:
        for b in total_budgets:
            if b >= F.n_terms(start): continue
            c = S.candidate(PPs[r], b); cands += list(c.values()); log(f'round {r}, {b} terms: validation {S.score(c[2]):.4f} / {S.score(c[3]):.4f}')
    chosen, info = S.choose(cands); return chosen, info
